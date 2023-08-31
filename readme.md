# adetailer_scripts
## Project structure
- configs
  - kinds of configuration files
- data 
  - some data for test
- log
  - log for procedure and requests
- pipelines
  - diffuser pipeline for adetailer
- scripts
  - process scripts
- test
  - test scripts
- utils
  - some tools for scripts
- server
  - kinds of api to handle request
## Execution process 
1. start api server
2. build two pipelines
2. request come
3. dispatch request to scripts
4. pipelines load loras according to the request content 
5. process images using pipelines
6. unload loras
## Usage
1. standard usage for adetailer without face feature was shown in `test/test_standard.py`
2. specific usage for adetailer with face feature was shown in `test/test_with_faceF.py`
## afterward optimization
[] log系统稍稍有点粗糙
[] 根据引导图bbox应对mask生成bad case进行处理
[] 增加同一角色不同lora切换的功能，根据request中给出的参数

