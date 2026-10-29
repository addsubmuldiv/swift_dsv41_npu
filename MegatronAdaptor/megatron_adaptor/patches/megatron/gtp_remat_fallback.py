# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Patch ``get_data_parallel_group`` to fall back to non-GTP groups when the
GTP_REMAT variant is not yet initialized.

Root cause: Megatron-LM commit c5ff22b7f (GTP #4967) changed
``get_data_parallel_group`` to default ``with_gtp_remat=True``.  The SHARP
barrier inside ``initialize_model_parallel()`` calls this function *before*
the GTP_REMAT group aliasing runs (when ``gtp_remat_size=1``), so
``_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT`` is still ``None`` → assertion
failure.

This is an **upstream Megatron-LM bug** — GPU users with ``--use-sharp``
will see the same crash.  The wrapper catches the ``AssertionError`` and
retries with ``with_gtp_remat=False`` so the SHARP barrier succeeds using
the already-initialized non-GTP group.  Once upstream fixes the ordering,
this patch and its CLI flag can be removed.

Usage: ``--fix-gtp-sharp-init-ordering`` (MegatronFixFeature).
"""


def get_data_parallel_group_gtp_fallback_wrapper(orig_fn):
    """Wrap ``get_data_parallel_group`` with GTP_REMAT fallback.

    When ``with_gtp_remat=True`` (the new default) and the GTP_REMAT group
    has not been aliased yet (``gtp_remat_size=1`` + SHARP barrier runs
    before the aliasing assignment), fall back to the non-GTP variant.
    """

    def wrapper(with_context_parallel=False, with_gtp_remat=True, partial_data_parallel=False):
        try:
            return orig_fn(
                with_context_parallel=with_context_parallel,
                with_gtp_remat=with_gtp_remat,
                partial_data_parallel=partial_data_parallel,
            )
        except AssertionError:
            if with_gtp_remat:
                return orig_fn(
                    with_context_parallel=with_context_parallel,
                    with_gtp_remat=False,
                    partial_data_parallel=partial_data_parallel,
                )
            raise

    return wrapper
