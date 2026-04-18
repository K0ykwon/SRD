"""Tests for the block-refresh detail-memory SRD variant."""

import torch

from srd.config import SRDConfig
from srd.modeling.block_refresh_detail_model import BlockRefreshDetailModel
from srd.modeling.block_refresh_model import BlockRefreshModel
from srd.training.losses import compute_srd_loss


def test_detail_model_outputs_expected_shapes() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh_detail",
        vocab_size=32,
        d_model=16,
        num_heads=4,
        block_size=4,
        segment_length=4,
        refresh_slots=2,
        refresh_count=2,
        detail_enabled=True,
        detail_slots=4,
        detail_topk=2,
        refresh_enabled=True,
        use_refresh=True,
    )
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["refresh_states"].shape == (2, 4, config.d_model)
    assert outputs["detail_states"].shape[2] == config.d_model
    assert outputs["debug"]["detail_enabled"] is True
    assert outputs["debug"]["detail_scan_carry_mode"] == "legacy"
    assert outputs["debug"]["detail_forward_mode"] == "sequential"
    assert outputs["debug"]["detail_coarse_group_size"] == 0
    assert outputs["debug"]["detail_coarse_topk_groups"] == 0
    assert outputs["debug"]["prefix_carry_state_shape"] == (2, 2, config.d_model)
    assert outputs["debug"]["fused_context_state_shape"] == (2, 2, config.d_model)


def test_detail_defaults_keep_original_execution_path() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    assert config.detail_forward_mode == "sequential"
    assert config.detail_coarse_group_size == 0
    assert config.detail_coarse_topk_groups == 0


def test_parallel_scan_detail_forward_preserves_output_surface() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_forward_mode = "parallel_scan"
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size * 3))

    outputs = model(input_ids)

    assert outputs["logits"].shape == (2, config.block_size * 3, config.vocab_size)
    assert outputs["refresh_states"].shape[2] == config.d_model
    assert outputs["detail_states"].shape[2] == config.d_model
    assert outputs["debug"]["detail_forward_mode"] == "parallel_scan"
    assert outputs["debug"]["refresh_bank_access_count"] == 0


def test_parallel_scan_detail_forward_backpropagates() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_forward_mode = "parallel_scan"
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size * 3))

    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    assert model.detail_query_proj.weight.grad is not None
    assert model.block_to_refresh.weight.grad is not None


def test_parallel_scan_rejects_grouped_detail_retrieval() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_forward_mode = "parallel_scan"
    config.detail_coarse_group_size = 2
    config.detail_coarse_topk_groups = 2
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size * 2))

    try:
        model(input_ids)
    except ValueError as exc:
        assert "grouped coarse retrieval" in str(exc)
    else:
        raise AssertionError("parallel_scan should reject grouped detail retrieval")


def test_detail_scan_reconstruction_matches_online_prefix_trace() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size * 3))

    embeddings = model.embedding(input_ids)
    block_embeddings, _, _ = model._reshape_blocks(embeddings)
    pre_hidden_blocks = model._encode_blocks_parallel(block_embeddings)
    precomputed_targets = model._precompute_next_block_targets(input_ids)
    bank_states = model.long_bank.empty(input_ids.size(0), input_ids.device)
    scan_outputs = model._scan_detail_block_sequence(pre_hidden_blocks, precomputed_targets, bank_states)

    assert torch.equal(scan_outputs["prefix_carry_mask"], scan_outputs["online_prefix_carry_mask"])
    mask = scan_outputs["prefix_carry_mask"].expand_as(scan_outputs["prefix_carry_states"])
    assert torch.allclose(
        scan_outputs["prefix_carry_states"][mask],
        scan_outputs["online_prefix_carry_states"][mask],
        atol=1e-5,
        rtol=1e-5,
    )

def test_online_detail_block_step_matches_full_step_hidden_shape() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))
    embeddings = model.embedding(input_ids)
    block_embeddings, _, _ = model._reshape_blocks(embeddings)
    pre_hidden_blocks = model._encode_blocks_parallel(block_embeddings)
    detail_key_blocks, detail_value_blocks = model._allocate_detail_kv_cache(2, config.detail_slots, input_ids.device)
    step_outputs = model._materialize_online_detail_block_step(
        pre_hidden_blocks[:, 0, :, :],
        None,
        detail_key_blocks,
        detail_value_blocks,
        0,
    )

    assert step_outputs["coarse_hidden_states"].shape == (2, config.block_size, config.d_model)
    assert step_outputs["pooled_hidden"].shape == (2, config.d_model)
    assert step_outputs["hidden_states"].shape == (2, config.block_size, config.d_model)


def test_detail_grouped_refinement_reduces_fine_candidates_without_dropping_history() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_coarse_group_size = 2
    config.detail_coarse_topk_groups = 2
    model = BlockRefreshDetailModel(config)
    pooled_hidden = torch.randn(1, config.d_model)
    total_candidates = config.detail_slots * 4
    detail_keys = torch.randn(1, total_candidates, config.d_model)
    detail_values = torch.randn(1, total_candidates, config.d_model)

    detail_context, trace = model._retrieve_detail_context(
        pooled_hidden,
        detail_keys,
        detail_values,
    )

    assert detail_context is not None
    assert trace["detail_candidate_count"] == total_candidates
    assert trace["detail_group_count"] == total_candidates // config.detail_coarse_group_size
    assert trace["detail_group_topk_used"] == config.detail_coarse_topk_groups
    assert trace["detail_fine_candidate_count"] == config.detail_coarse_group_size * config.detail_coarse_topk_groups
    assert trace["detail_topk_used"] == config.detail_topk


def test_detail_slots_selection_respects_capacity() -> None:
    config = SRDConfig(
        model_type="srd_block_refresh_detail",
        d_model=8,
        num_heads=2,
        detail_enabled=True,
        detail_slots=3,
        detail_topk=2,
        detail_saliency_slots=2,
    )
    model = BlockRefreshDetailModel(config)
    hidden_states = torch.randn(2, 5, config.d_model)
    detail_slots, positions = model._select_detail_slots(hidden_states)

    assert detail_slots.shape[1] <= 3
    assert positions.shape[1] == detail_slots.shape[1]


def test_detail_variant_preserves_base_refresh_param_count_plus_small_overhead() -> None:
    base = BlockRefreshModel(SRDConfig.preset("block_refresh_suf_tiny"))
    detail = BlockRefreshDetailModel(SRDConfig.preset("block_refresh_detail_tiny"))
    base_params = sum(parameter.numel() for parameter in base.parameters())
    detail_params = sum(parameter.numel() for parameter in detail.parameters())

    assert detail_params > base_params
    assert detail_params - base_params < 25000


def test_detail_retrieval_affects_later_blocks_only() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_a = torch.randint(0, config.vocab_size, (1, 16))
    input_b = input_a.clone()
    input_b[:, : config.block_size] = (input_b[:, : config.block_size] + 1) % config.vocab_size

    outputs_a = model(input_a)
    outputs_b = model(input_b)

    assert not torch.allclose(outputs_a["logits"][:, config.block_size :, :], outputs_b["logits"][:, config.block_size :, :])
    assert outputs_a["debug"]["detail_access_count"] >= 0


def test_detail_variant_sufficiency_loss_backpropagates() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    assert losses["sufficiency_loss"].item() >= 0.0
    assert model.sufficiency_head.weight.grad is not None


def test_detail_prefill_decode_matches_padded_forward_logits() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    model.eval()
    full_prefix = torch.randint(0, config.vocab_size, (1, config.block_size))
    full_outputs = model(full_prefix)
    full_state = model.prefill(full_prefix)
    assert torch.allclose(full_state["next_logits"], full_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)
    assert full_state["completed_prefix_carry_states"].shape == (1, 1, config.d_model)
    assert full_state["completed_fused_context_states"].shape == (1, 1, config.d_model)

    completed_embeddings = model.embedding(full_prefix)
    completed_block_embeddings, _, _ = model._reshape_blocks(completed_embeddings)
    completed_pre_hidden_blocks = model._encode_blocks_parallel(completed_block_embeddings)
    detail_key_blocks, detail_value_blocks = model._allocate_detail_kv_cache(1, config.detail_slots, full_prefix.device)
    completed_outputs = model._scan_completed_blocks_prefill(
        completed_pre_hidden_blocks,
        model.long_bank.empty(1, full_prefix.device),
        detail_key_blocks,
        detail_value_blocks,
        0,
    )
    assert torch.equal(
        completed_outputs["prefix_carry_mask"],
        completed_outputs["online_prefix_carry_mask"],
    )

    partial_prefix = torch.randint(0, config.vocab_size, (1, config.block_size - 1))
    partial_state = model.prefill(partial_prefix)
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    stepped = model.decode_step(next_token, partial_state)
    completed = torch.cat([partial_prefix, next_token], dim=1)
    completed_outputs = model(completed)
    assert torch.allclose(stepped["next_logits"], completed_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)


def test_detail_prefill_decode_matches_forward_after_multiple_steps() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    model = BlockRefreshDetailModel(config)
    model.eval()

    partial_prefix = torch.randint(0, config.vocab_size, (1, config.block_size - 2))
    next_tokens = torch.randint(0, config.vocab_size, (1, 2))
    state = model.prefill(partial_prefix)
    state = model.decode_step(next_tokens[:, :1], state)
    state = model.decode_step(next_tokens[:, 1:], state)

    completed = torch.cat([partial_prefix, next_tokens], dim=1)
    completed_outputs = model(completed)
    assert torch.allclose(state["next_logits"], completed_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)


def test_affine_detail_scan_carry_preserves_forward_decode_consistency() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_scan_carry_mode = "affine"
    model = BlockRefreshDetailModel(config)
    model.eval()

    full_prefix = torch.randint(0, config.vocab_size, (1, config.block_size))
    full_outputs = model(full_prefix)
    full_state = model.prefill(full_prefix)
    assert full_outputs["debug"]["detail_scan_carry_mode"] == "affine"
    assert torch.allclose(full_state["next_logits"], full_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)

    partial_prefix = torch.randint(0, config.vocab_size, (1, config.block_size - 2))
    next_tokens = torch.randint(0, config.vocab_size, (1, 2))
    state = model.prefill(partial_prefix)
    state = model.decode_step(next_tokens[:, :1], state)
    state = model.decode_step(next_tokens[:, 1:], state)

    completed = torch.cat([partial_prefix, next_tokens], dim=1)
    completed_outputs = model(completed)
    assert torch.allclose(state["next_logits"], completed_outputs["logits"][:, -1, :], atol=1e-5, rtol=1e-5)


def test_affine_detail_scan_carry_receives_gradient_on_three_blocks() -> None:
    config = SRDConfig.preset("block_refresh_detail_tiny")
    config.detail_scan_carry_mode = "affine"
    model = BlockRefreshDetailModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size * 3))
    outputs = model(input_ids)
    losses = compute_srd_loss(outputs, input_ids, config)
    losses["loss"].backward()

    assert model.scan_carry_gate is not None
    assert model.scan_carry_gate[0].weight.grad is not None
