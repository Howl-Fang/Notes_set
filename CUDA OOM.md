原因：没有使用多卡

<details>
<summary>Question</summary>

(EngineCore_DP0 pid=2273233) INFO 02-28 17:42:14 [gpu_model_runner.py:5140] Encoder cache will be initialized with a budget of 18605 tokens, and profiled with 1 video items of the maximum feature size.
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006] EngineCore failed to start.
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006] Traceback (most recent call last):
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 996, in run_engine_core
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 740, in __init__
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     super().__init__(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 113, in __init__
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 248, in _initialize_kv_caches
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 128, in determine_available_memory
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self.collective_rpc("determine_available_memory")
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 75, in collective_rpc
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return func(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return func(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 339, in determine_available_memory
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     self.model_runner.profile_run()
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 5156, in profile_run
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     dummy_encoder_outputs = self.model.embed_multimodal(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 1579, in embed_multimodal
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     video_embeddings = self._process_video_input(multimodal_input)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 1526, in _process_video_input
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return run_dp_sharded_mrope_vision_model(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/vision.py", line 494, in run_dp_sharded_mrope_vision_model
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     image_embeds_local = vision_model(pixel_values_local, local_grid_thw_list)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 767, in forward
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     x = blk(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]         ^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 405, in forward
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     x = residual + self.mlp(x_fused_norm)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                    ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 224, in forward
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     x = self.act_fn(x)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]         ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/custom_op.py", line 126, in forward
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self._forward_method(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 832, in compile_wrapper
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return fn(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py", line 137, in forward_native
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     @staticmethod
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1044, in _fn
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return fn(*args, **kwargs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 1130, in forward
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return compiled_fn(full_args)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 353, in runtime_wrapper
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     all_outs = call_func_at_runtime_with_args(
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py", line 129, in call_func_at_runtime_with_args
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     out = normalize_as_list(f(args))
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]                             ^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 724, in inner_fn
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     outs = compiled_fn(args)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 526, in wrapper
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return compiled_fn(runtime_args)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/output_code.py", line 613, in __call__
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     return self.current_callable(inputs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py", line 3017, in run
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     out = model(new_inputs)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]           ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]   File "/tmp/torchinductor_hligi/ha/chavmznvmjpfllttktjuaogwparltca35vsbxshitafspr6q662w.py", line 100, in call
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]     buf0 = empty_strided_cuda((s77, 1, s53 // 2), (max(1, s53 // 2), max(1, s53 // 2), 1), torch.bfloat16)
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) ERROR 02-28 17:42:17 [core.py:1006] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 698.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 559.00 MiB is free. Including non-PyTorch memory, this process has 23.01 GiB memory in use. Of the allocated memory 22.08 GiB is allocated by PyTorch, and 614.16 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables  )
(EngineCore_DP0 pid=2273233) Process EngineCore_DP0:
(EngineCore_DP0 pid=2273233) Traceback (most recent call last):
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_DP0 pid=2273233)     self.run()
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore_DP0 pid=2273233)     self._target(*self._args, **self._kwargs)
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1010, in run_engine_core
(EngineCore_DP0 pid=2273233)     raise e
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 996, in run_engine_core
(EngineCore_DP0 pid=2273233)     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore_DP0 pid=2273233)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 740, in __init__
(EngineCore_DP0 pid=2273233)     super().__init__(
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 113, in __init__
(EngineCore_DP0 pid=2273233)     num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
(EngineCore_DP0 pid=2273233)                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 248, in _initialize_kv_caches
(EngineCore_DP0 pid=2273233)     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore_DP0 pid=2273233)                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 128, in determine_available_memory
(EngineCore_DP0 pid=2273233)     return self.collective_rpc("determine_available_memory")
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 75, in collective_rpc
(EngineCore_DP0 pid=2273233)     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_DP0 pid=2273233)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 459, in run_method
(EngineCore_DP0 pid=2273233)     return func(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=2273233)     return func(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 339, in determine_available_memory
(EngineCore_DP0 pid=2273233)     self.model_runner.profile_run()
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 5156, in profile_run
(EngineCore_DP0 pid=2273233)     dummy_encoder_outputs = self.model.embed_multimodal(
(EngineCore_DP0 pid=2273233)                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 1579, in embed_multimodal
(EngineCore_DP0 pid=2273233)     video_embeddings = self._process_video_input(multimodal_input)
(EngineCore_DP0 pid=2273233)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 1526, in _process_video_input
(EngineCore_DP0 pid=2273233)     return run_dp_sharded_mrope_vision_model(
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/vision.py", line 494, in run_dp_sharded_mrope_vision_model
(EngineCore_DP0 pid=2273233)     image_embeds_local = vision_model(pixel_values_local, local_grid_thw_list)
(EngineCore_DP0 pid=2273233)                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233)     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233)     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 767, in forward
(EngineCore_DP0 pid=2273233)     x = blk(
(EngineCore_DP0 pid=2273233)         ^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233)     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233)     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 405, in forward
(EngineCore_DP0 pid=2273233)     x = residual + self.mlp(x_fused_norm)
(EngineCore_DP0 pid=2273233)                    ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233)     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233)     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/models/glm4_1v.py", line 224, in forward
(EngineCore_DP0 pid=2273233)     x = self.act_fn(x)
(EngineCore_DP0 pid=2273233)         ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=2273233)     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=2273233)     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/custom_op.py", line 126, in forward
(EngineCore_DP0 pid=2273233)     return self._forward_method(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 832, in compile_wrapper
(EngineCore_DP0 pid=2273233)     return fn(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/activation.py", line 137, in forward_native
(EngineCore_DP0 pid=2273233)     @staticmethod
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1044, in _fn
(EngineCore_DP0 pid=2273233)     return fn(*args, **kwargs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py", line 1130, in forward
(EngineCore_DP0 pid=2273233)     return compiled_fn(full_args)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 353, in runtime_wrapper
(EngineCore_DP0 pid=2273233)     all_outs = call_func_at_runtime_with_args(
(EngineCore_DP0 pid=2273233)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py", line 129, in call_func_at_runtime_with_args
(EngineCore_DP0 pid=2273233)     out = normalize_as_list(f(args))
(EngineCore_DP0 pid=2273233)                             ^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 724, in inner_fn
(EngineCore_DP0 pid=2273233)     outs = compiled_fn(args)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 526, in wrapper
(EngineCore_DP0 pid=2273233)     return compiled_fn(runtime_args)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/output_code.py", line 613, in __call__
(EngineCore_DP0 pid=2273233)     return self.current_callable(inputs)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py", line 3017, in run
(EngineCore_DP0 pid=2273233)     out = model(new_inputs)
(EngineCore_DP0 pid=2273233)           ^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233)   File "/tmp/torchinductor_hligi/ha/chavmznvmjpfllttktjuaogwparltca35vsbxshitafspr6q662w.py", line 100, in call
(EngineCore_DP0 pid=2273233)     buf0 = empty_strided_cuda((s77, 1, s53 // 2), (max(1, s53 // 2), max(1, s53 // 2), 1), torch.bfloat16)
(EngineCore_DP0 pid=2273233)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2273233) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 698.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 559.00 MiB is free. Including non-PyTorch memory, this process has 23.01 GiB memory in use. Of the allocated memory 22.08 GiB is allocated by PyTorch, and 614.16 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables  )
[rank0]:[W228 17:42:18.288777459 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown   (function operator())
(APIServer pid=2272936) Traceback (most recent call last):
(APIServer pid=2272936)   File "frozen runpy", line 198, in _run_module_as_main
(APIServer pid=2272936)   File "frozen runpy", line 88, in _run_code
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 531, in module
(APIServer pid=2272936)     uvloop.run(run_server(args))
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=2272936)     return __asyncio.run(
(APIServer pid=2272936)            ^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/runners.py", line 195, in run
(APIServer pid=2272936)     return runner.run(main)
(APIServer pid=2272936)            ^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=2272936)     return self._loop.run_until_complete(task)
(APIServer pid=2272936)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=2272936)     return await main
(APIServer pid=2272936)            ^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 457, in run_server
(APIServer pid=2272936)     await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 476, in run_server_worker
(APIServer pid=2272936)     async with build_async_engine_client(
(APIServer pid=2272936)                ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=2272936)     return await anext(self.gen)
(APIServer pid=2272936)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 96, in build_async_engine_client
(APIServer pid=2272936)     async with build_async_engine_client_from_engine_args(
(APIServer pid=2272936)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=2272936)     return await anext(self.gen)
(APIServer pid=2272936)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 137, in build_async_engine_client_from_engine_args
(APIServer pid=2272936)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=2272936)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 222, in from_vllm_config
(APIServer pid=2272936)     return cls(
(APIServer pid=2272936)            ^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 148, in __init__
(APIServer pid=2272936)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=2272936)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 124, in make_async_mp_client
(APIServer pid=2272936)     return AsyncMPClient(*client_args)
(APIServer pid=2272936)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 835, in __init__
(APIServer pid=2272936)     super().__init__(
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 490, in __init__
(APIServer pid=2272936)     with launch_core_engines(vllm_config, executor_class, log_stats) as (
(APIServer pid=2272936)          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2272936)   File "/localdata/hf/vllm/tmp/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=2272936)     next(self.gen)
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 925, in launch_core_engines
(APIServer pid=2272936)     wait_for_engine_startup(
(APIServer pid=2272936)   File "/localdata/hf/vllm/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 984, in wait_for_engine_startup
(APIServer pid=2272936)     raise RuntimeError(
(APIServer pid=2272936) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
这是新产生的错误
</details>

在 vLLM 中利用多张显卡主要有两种方式：**Tensor Parallelism (TP)** 和 **Data Parallelism (DP)**。针对你的 `AutoGLM-Phone-9B` 多模态模型，以下是完整配置指南：

---

## 🔑 核心概念

| 并行方式 | 作用 | 适用场景 | 显存效果 |
|---------|------|---------|---------|
| **TP (Tensor Parallel)** | 将模型层切分到多卡 | 单模型太大，单卡放不下 | ✅ 显存线性分摊 |
| **DP (Data Parallel)** | 多卡运行相同模型，处理不同请求 | 高并发推理 | ❌ 每卡仍需完整模型 |

> 💡 **9B 多模态模型推荐**：优先使用 **TP=2**，2 张卡各承担 ~12GB 模型权重 + 编码器，轻松跑满 24G 显存。

---

## 🚀 多卡启动配置

### 方案 A：Tensor Parallelism（推荐 ⭐）

```bash
# 假设你有 2 张 GPU (0,1)
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name autoglm-phone-9b \
  --model zai-org/AutoGLM-Phone-9B \
  --port 8000 \
  \
  --tensor-parallel-size 2 \                    # 🔑 TP=2，模型切分到2卡 \
  --mm-encoder-tp-mode data \                   # 视频编码器使用 DP 模式（避免 TP 切分编码器） \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\": 2000000}" \
  --max-model-len 20480 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\": 4}" \
  \
  --gpu-memory-utilization 0.90 \               # TP 模式下可适当提高 \
  --enforce-eager \                             # 多卡时建议启用，减少同步开销 \
  --distributed-executor-backend ray            # 或使用 'mp' (multiprocessing)
```

### 方案 B：Data Parallelism（高并发场景）

```bash
# DP 需要 vLLM v0.6.0+ 且配置更复杂，通常配合 TP 使用
# 示例：2 节点 × 2 卡 = 4 实例，每实例 TP=1
export CUDA_VISIBLE_DEVICES=0  # 实例1
python3 -m vllm.entrypoints.openai.api_server --port 8000 --tensor-parallel-size 1 ... &

export CUDA_VISIBLE_DEVICES=1  # 实例2  
python3 -m vllm.entrypoints.openai.api_server --port 8001 --tensor-parallel-size 1 ... &
# 前端用 Nginx/负载均衡器分发请求到 8000/8001
```

---

## ⚙️ 关键参数详解

```bash
# 1. 基础并行
--tensor-parallel-size 2          # 使用 2 张卡做模型并行（最常用）
--pipeline-parallel-size 1        # PP 通常设为 1，除非模型极大 (>70B)

# 2. 多模态特殊配置（AutoGLM 重点！）
--mm-encoder-tp-mode data         # ✅ 编码器不切分，每卡独立处理（兼容性好）
# --mm-encoder-tp-mode tensor     # ❌ 尝试切分编码器（可能报错，需模型支持）

--mm_processor_cache_type shm     # 多卡共享内存缓存，减少重复加载

# 3. 分布式后端
--distributed-executor-backend ray   # 推荐：Ray 管理多进程更稳定
# --distributed-executor-backend mp  # 备选：multiprocessing（简单场景）

# 4. 显存优化（多卡仍需注意）
--gpu-memory-utilization 0.90   # TP 模式下可设更高（0.85~0.92）
--kv-cache-dtype fp8            # 如硬件支持，KV cache 用 fp8 省 50% 显存
```

---

## 🔍 验证多卡是否生效

启动后观察日志：
```log
INFO ... World size: 2, Tensor parallel size: 2    # ✅ TP=2 生效
INFO ... # GPU blocks: 1234 per GPU                # ✅ 每卡独立 block 数
INFO ... Distributed execution with backend: ray   # ✅ 后端正常
```

或用命令检查：
```bash
# 查看进程占用 GPU
nvidia-smi pmon -c 1

# 查看 NCCL 通信（TP 依赖）
nccl-debug=info python3 -m vllm.entrypoints.openai.api_server ... 2>&1 | grep NCCL
```

---

## 🛠️ 常见问题排查

### ❌ 问题 1：`NCCL connection error` / `connection refused`
```bash
# 解决方案：
export NCCL_SOCKET_IFNAME=eth0          # 指定网卡（多机时必需）
export NCCL_IB_DISABLE=1                # 无 InfiniBand 时禁用
export NCCL_DEBUG=INFO                  # 查看详细通信日志
```

### ❌ 问题 2：多卡但显存没分摊（仍 OOM）
```bash
# 检查点：
# 1. 确认 --tensor-parallel-size 已设置
# 2. 确认 CUDA_VISIBLE_DEVICES 包含多卡
# 3. 多模态编码器可能未切分：尝试 --mm-encoder-tp-mode tensor（如模型支持）
# 4. 降低 max_pixels / max-model-len 先验证 TP 基础功能
```

### ❌ 问题 3：Ray 启动失败
```bash
# 手动初始化 Ray（调试用）：
ray start --head --port=6379 --num-gpus=2
# 然后在 vLLM 命令中加：
--ray-address=auto
```

---

## 📊 不同 GPU 数量推荐配置

| GPU 数量 | 推荐 TP | 显存收益 | 备注 |
|---------|---------|---------|------|
| 1×24G | TP=1 | 基准 | 需严格限制 `max_pixels≤1M` |
| **2×24G** | **TP=2** | ✅ 模型权重减半 | ⭐ 最佳性价比，推荐 |
| 4×24G | TP=2 + DP=2 | ✅ 显存+吞吐双提升 | 需配置负载均衡 |
| 2×48G (A100) | TP=2 | ✅ 可开更大 context | 适合长视频理解 |

---

## 🎯 最终推荐命令（2 卡 24G）

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARNING  # 生产环境设为 WARNING 减少日志

python3 -m vllm.entrypoints.openai.api_server \
  --model zai-org/AutoGLM-Phone-9B \
  --served-model-name autoglm-phone-9b \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\": 2000000, \"max_frames\": 16}" \
  --max-model-len 20480 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\": 4, \"video\": 1}" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --distributed-executor-backend ray \
  --host 0.0.0.0
```

> ✅ 此配置下：  
> - 模型权重 ~9GB → 每卡 ~4.5GB  
> - 视频编码器 ~3GB/卡  
> - KV cache + 激活值 ~10GB/卡  
> - **总计 ~17.5GB/卡 < 24GB**，稳定运行 ✅

如果仍有问题，请提供：
1. `nvidia-smi` 输出（GPU 型号/数量）
2. `vllm` 版本：`pip show vllm | grep Version`
3. 完整启动日志前 50 行

我可以帮你进一步调优！🔧

>现在我用的命令是:
>python3 -m vllm.entrypoints.openai.api_server \
  --served-model-name autoglm-phone-9b \
  --allowed-local-media-path / \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\":5000000}" \
  --max-model-len 25480 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\":10}" \
  --model zai-org/AutoGLM-Phone-9B \
  --port 8888

我总共有8张3090，帮我修改命令

针对你 **8 张 RTX 3090 (24GB)** 的配置，运行 `zai-org/AutoGLM-Phone-9B` 多模态模型，我为你提供两套方案：

---

## 🎯 方案一：单实例 TP=4（推荐 ⭐ 稳定优先）

使用 4 张卡做 **Tensor Parallelism**，模型权重分摊，显存压力最小，适合生产环境。

```bash
#!/bin/bash
# 使用前 4 张卡 (0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARNING
export RAY_DEDUP_LOGS=1

python3 -m vllm.entrypoints.openai.api_server \
  --model zai-org/AutoGLM-Phone-9B \
  --served-model-name autoglm-phone-9b \
  --port 8888 \
  --host 0.0.0.0 \
  \
  --tensor-parallel-size 4 \                           # 🔑 4 卡模型并行 \
  --mm-encoder-tp-mode data \                          # 编码器每卡独立加载（兼容性好） \
  --mm_processor_cache_type shm \                      # 多卡共享内存缓存 \
  --mm_processor_kwargs "{\"max_pixels\": 2000000, \"max_frames\": 16}" \  # 500 万→200 万像素 \
  --max-model-len 20480 \                              # 25480→20480，保守显存 \
  --chat-template-content-format string \
  --limit-mm-per-prompt "{\"image\": 4, \"video\": 1}" \  # 10 张→4 张图片并发 \
  \
  --gpu-memory-utilization 0.88 \                      # 预留 12% 给编码器峰值 \
  --enforce-eager \                                    # 多卡时减少内存碎片 \
  --distributed-executor-backend ray \                 # Ray 管理多进程 \
  --enable-prefix-caching \                            # 开启 prompt 缓存（提升多轮对话效率）\
  --swap-space 4                                       # 4GB CPU 交换空间（防 OOM 缓冲）
```

### ✅ 显存估算（TP=4，每卡 24GB）
| 组件 | 每卡占用 | 说明 |
|------|---------|------|
| 模型权重 (BF16) | ~4.5 GB | 9B / 4 卡 |
| 视频编码器 | ~3.5 GB | data mode 下每卡独立加载 |
| KV Cache + 激活 | ~10-12 GB | max-model-len=20480 |
| **总计** | **~18-20 GB** | ✅ 24GB 显存内稳定运行 |

---

## 🚀 方案二：8 卡全用（高吞吐 DP+TP）

启动 **2 个独立实例**，每个 TP=4，分别绑定不同 GPU，前端用 Nginx 负载均衡。

### 实例 1（GPU 0-3，端口 8888）
```bash
# start_instance_1.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m vllm.entrypoints.openai.api_server \
  --model zai-org/AutoGLM-Phone-9B \
  --served-model-name autoglm-phone-9b \
  --port 8888 \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\": 2000000}" \
  --max-model-len 20480 \
  --limit-mm-per-prompt "{\"image\": 4}" \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --distributed-executor-backend ray \
  --ray-namespace autoglm_instance1 &  # 隔离 Ray 命名空间
```

### 实例 2（GPU 4-7，端口 8889）
```bash
# start_instance_2.sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m vllm.entrypoints.openai.api_server \
  --model zai-org/AutoGLM-Phone-9B \
  --served-model-name autoglm-phone-9b \
  --port 8889 \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --mm_processor_cache_type shm \
  --mm_processor_kwargs "{\"max_pixels\": 2000000}" \
  --max-model-len 20480 \
  --limit-mm-per-prompt "{\"image\": 4}" \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --distributed-executor-backend ray \
  --ray-namespace autoglm_instance2 &
```

### 🔁 Nginx 负载均衡配置（可选）
```nginx
# /etc/nginx/conf.d/autoglm.conf
upstream autoglm_backend {
    least_conn;  # 最少连接数调度
    server 127.0.0.1:8888;
    server 127.0.0.1:8889;
}

server {
    listen 80;
    location / {
        proxy_pass http://autoglm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 100M;  # 支持大图片/视频上传
    }
}
```

---

## ⚙️ 参数调优建议（3090 专属）

```bash
# 🔸 多模态参数（关键！3090 显存有限）
--mm_processor_kwargs "{\"max_pixels\": 2000000, \"max_frames\": 16, \"min_pixels\": 256000}"
# max_pixels: 200 万 ≈ 1920×1080，平衡画质与显存
# max_frames: 视频最多采样 16 帧，避免长视频 OOM

# 🔸 并发控制
--limit-mm-per-prompt "{\"image\": 4, \"video\": 1}"
--max-num-seqs 128  # 单实例最大并发请求数（3090 建议 64~128）

# 🔸 显存优化
--gpu-memory-utilization 0.88      # 3090 建议 0.85~0.90，留余量给编码器
--kv-cache-dtype fp8              # ✅ 如 vLLM+PyTorch 支持，KV cache 省 50% 显存
--enforce-eager                   # 多卡时禁用 CUDA Graph，减少碎片

# 🔸 分布式通信优化（8 卡/多机必备）
export NCCL_SOCKET_IFNAME=eth0           # 指定通信网卡
export NCCL_IB_DISABLE=1                 # 无 InfiniBand 时禁用
export NCCL_MIN_NRINGS=4                 # 增加通信环数，提升 TP 效率
```

---

## 🔍 启动后验证

```bash
# 1. 检查 GPU 占用
watch -n 1 nvidia-smi

# 2. 查看 vLLM 日志关键行
# ✅ 应看到：
# "World size: 4, Tensor parallel size: 4"
# "# GPU blocks: XXX per GPU"
# "Loading model ... on 4 GPUs"

# 3. 测试 API
curl http://localhost:8888/v1/models  # 查看模型是否就绪

# 4. 压力测试（可选）
# 用 locust 或 ab 测试 QPS，观察显存/延迟变化
```

---

## 🛠️ 常见问题速查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `NCCL unhandled system error` | 网卡/PCIe 通信问题 | `export NCCL_SOCKET_IFNAME=eth0` + 检查 `dmesg \| grep NVRM` |
| 多卡但显存没分摊 | `--tensor-parallel-size` 未生效 | 确认命令含 `--tensor-parallel-size 4` + `CUDA_VISIBLE_DEVICES` |
| 视频推理 OOM | 编码器峰值超限 | 降低 `max_pixels` / `max_frames` + 加 `--swap-space 4` |
| Ray 启动卡住 | 端口冲突/权限 | `ray stop --force` 清理 + 用 `--ray-namespace` 隔离 |

---

## 📊 8×3090 配置推荐总结

| 目标 | 推荐配置 | 预期效果 |
|------|---------|---------|
| ✅ **稳定运行** | 单实例 TP=4（方案一） | 单请求延迟低，显存充裕，适合 API 服务 |
| 🚀 **高吞吐** | 双实例 TP=4 + Nginx（方案二） | QPS ×2，适合批量推理/高并发场景 |
| 🔬 **实验调试** | TP=2 + 大 context | 用 2 卡跑 `max-model-len=32768`，研究长视频理解 |

> 💡 **经验建议**：先用**方案一**验证模型能稳定加载 + 推理，再扩展到方案二。AutoGLM-Phone 的视频编码器对显存敏感，**宁可保守参数，避免反复 OOM 重启**。

如果需要我帮你：
1. 写一个自动监控显存 + 重启的守护脚本？
2. 配置 Prometheus + Grafana 监控 vLLM 指标？
3. 针对纯图片/纯视频场景进一步优化参数？

随时告诉我！🔧