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