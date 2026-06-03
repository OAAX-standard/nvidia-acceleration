def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--input-path', required=True,
                        help='Path to the input zip archive (must contain model.onnx + config.json; '
                             'int8 also requires a calib/ directory of images)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    import os
    import platform
    import tempfile
    from .utils import unzip_input, validate_config, check_nvidia_dependencies, detect_gpu_architecture, run_trtexec, build_fp16_engine, build_int8_engine, build_int8_engine_trt11, md5_hash, build_mime_type
    from .logger import Logs

    logs = Logs()
    logs.add_message('Starting TensorRT conversion', {'input_path': args.input_path})

    gpu_arch = detect_gpu_architecture()
    logs.add_message('Detected GPU architecture', {'gpu_architecture': gpu_arch})

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path, config = unzip_input(args.input_path, tmp_dir)

        logs.add_message('Input unpacked', {
            'onnx_md5': md5_hash(onnx_path),
            'config': config,
        })

        validate_config(config)
        logs.add_message('Config validated', config)

        trt_major = check_nvidia_dependencies(config['precision'])

        output_trt_path = os.path.join(args.output_dir, 'model.trt')

        if config['precision'] == 'int8':
            calib_dir = os.path.join(tmp_dir, config['calibration_data'].rstrip('/'))
            if not os.path.isdir(calib_dir):
                raise FileNotFoundError(
                    f"Calibration directory '{config['calibration_data']}' "
                    f"not found in zip archive")
            if trt_major is not None and trt_major >= 11:
                build_int8_engine_trt11(onnx_path, output_trt_path, calib_dir, config, logs)
            else:
                build_int8_engine(onnx_path, output_trt_path, calib_dir, config, logs)
        elif config['precision'] == 'fp16' and trt_major is not None and trt_major >= 11:
            build_fp16_engine(onnx_path, output_trt_path, config, logs)
        else:
            run_trtexec(onnx_path, output_trt_path, config, logs)

    mime_type = build_mime_type(gpu_arch, platform.machine())
    logs_path = os.path.join(args.output_dir, 'logs.json')

    logs.add_message('Successful Conversion', {
        'Output Directory': args.output_dir,
        'Output file name': 'model.trt',
        'MIME type': mime_type,
        'Output file MD5': md5_hash(output_trt_path),
        'Logs file name': 'logs.json',
    })

    logs.save_as_json(logs_path)
    print(logs)
    print('Exiting.')
