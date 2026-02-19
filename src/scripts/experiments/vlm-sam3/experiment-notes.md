input pairs is list of (aria_rgb_path, aria_mask_path, cam_rgb_path, cam_mask_path) tuples

processing: 
1. prints masks over object of interest
2. runs all experiments
3. stores metrics following sheets configuration

output: save CSV with results locally


--

Write a script called experiment.py that executes notebooks/VLM_SAM3_Experiments.py as an efficient experiment pipeline for much more takes. Below are details about the specifications of this file. Before beginning the development, ask me clarification questions.

* file ran with parser argument --config config.json
* script reads which files to look at via {split}_{direction}_pairs.json files, stored in root directory
* script initiates a big df to store all results, for all takes, for all frames 
* for each take, the script goes through the folder of the take and runs the exepriments iteratively
* input to pipeline for each take, for each frame: tuples structured as (aria_rgb_path, aria_mask_path, cam_rgb_path, cam_mask_path) 
* processing:
    0. decode mask; add TODO on top of .py file saying that this should be made efficient by extracting a priori the masks in the root folder using the structure we saw before in ego-exo correspondence. 
    1. add masks over object of interest to create source image + mask on target object
    2. run all experiments
    3. stores metrics on df
    4. after processing all pairs, computes summary metrics.
* output: summary metrics in csv format
* meaning of variables in config.json:
    * "root-data-directory" : "...", is the directory of where the data is found. for the virtual machine where this will be run, it is at "/data/video_datasets/3321908/output_dir_all/"
    * "vlm-api-key" : "...", is the api key of the vlm accessed using ollama
    "huggingface-api-key" : "...", is the api of huggingface to access sam 3 model
    * "vlm-model" : "...", is the vlm model to use to generate descriptions, name must be exactly the name ollama expects to work,
    * "huggingface-model" : "...", is sam3 version, name must be exactly what huggingface api expects
    "vlm-prompt" : "...", is the promt for the vlm telling to identify the object highlighterd by the mask and provide a descrition
    * "direction" : "egoexo", indicates direction of correspndence, used for selection of pairs
    * "subset-run-percentage" : 0.1, parameter to reduce the number of pairs on which the exepriemnts are run, used to debug and do tests before final runs
    * "subset-seed" : 42, seed for subset selection
* root directory structure: folder containing
    * takes, identified by UIDs like 01d7a29d-f1a2-44d6-8871-178f26233d32
    * split.json contains data split at take level via takes UIDs
    * {split}_{direction}_pairs.json indicate have directories for the pairs of data 
* notebook has many useless / repeating parts; we don't want them in the script; the script is concise and has only the code that it needs

--

I've thoroughly reviewed the notebook, config, data processing scripts, and mask loading utilities. Here are my clarification questions before I start building `experiment.py`:

---

**1. Structure of `{split}_{direction}_pairs.json`**

These files don't exist yet in the repo. What is the exact schema of each entry? Is it:
- (a) A flat list of 4-tuples with file paths: `[["aria_rgb.jpg", "aria_mask.png", "cam_rgb.jpg", "cam_mask.png"], ...]`
- (b) A structured list with metadata like `{"take_uid": "...", "object": "...", "aria_cam": "...", "exo_cam": "...", "frame": "..."}` from which paths are constructed?
- (c) Something else?

These files live on a virtual machine. (c) the pairs are directories that go inside the respective take folder: example `"/data/video_datasets/3321908/output_dir_all//41e70872-13dc-45e3-82b4-4cbe58d952da//aria01_214-1//egg nylon_0//rgb//1260", "/data/video_datasets/3321908/output_dir_all//41e70872-13dc-45e3-82b4-4cbe58d952da//aria01_214-1//egg nylon_0//mask//1260", "/data/video_datasets/3321908/output_dir_all//41e70872-13dc-45e3-82b4-4cbe58d952da//cam01//egg nylon_0//rgb//1260", "/data/video_datasets/3321908/output_dir_all//41e70872-13dc-45e3-82b4-4cbe58d952da//cam01//egg nylon_0//mask//1260"]`

**2. Mask source: files on disk vs. annotation.json decoding?**

The spec says input tuples include `aria_mask_path` and `cam_mask_path`, implying pre-existing mask image files. But in the notebook, masks are decoded at runtime from COCO RLE in `annotation.json` (originally LZString-compressed, pre-decoded by `process_data.py`). Which approach should the script use?
- If masks are decoded at runtime from `annotation.json`, what does the "mask path" in the tuple actually reference?
- Is the TODO about "extracting masks a priori" meant for a future optimization step, and for now we still decode from `annotation.json`?

The script should get the encoded mask and decode at runtime from COCO RLE in `annotation.json` (originally LZString-compressed, pre-decoded by `process_data.py`). As such, the TODO is meant for future optimisation. 

**3. Direction semantics**

When `direction = "egoexo"` (or `"ego-exo"` as in the current config), the VLM sees the **ego (aria)** image with overlay and SAM3 segments on the **exo (cam)** image — is that correct? Could `direction` also be `"exoego"` (reversing source/target)?

That is correct. The idea of the experiments is to give different information to the VLMs and see which generates the best prompts for SAM3. Possible directions are egoexo and exoego. 

**4. VLM selection: single or multiple per run?**

The notebook tests both Gemini and Qwen sequentially. The config has a single `"vlm-model"`. Should the script:
- (a) Run one VLM per invocation (determined by config), or
- (b) Support running multiple VLMs in one go?

(a) Run one VLM per invocation deterined by config.

**5. Timestamps (src_time / dest_time)**

The notebook hardcodes `src_time='0'` and `dest_time='300'`. Should the experiment script:
- (a) Keep these hardcoded?
- (b) Read them from the pairs file?
- (c) Make them configurable in `config.json`?

(b) The script reads them from the pairs file.

**6. Which splits to process?**

Should the script run on all splits (train, val, test) sequentially, or should the target split be a config/CLI parameter?

Focus on specific split of pairs. This is specified has to be secified in the config. 

**7. `src-prompt` vs `vlm-prompt`**

The config has both `"src-prompt"` and `"vlm-prompt"`. In the notebook, both are passed to the caption functions. Should both always be sent to the VLM together (concatenated), or is `src-prompt` optional / scenario-dependent?

Just use the VLM prompt. Each experiment will receive a different prompt. Therefore, the config needs to have 3 prompts, one for each experiment. To clarify, below are the experiments to be run, with their associated identification codes:
* EXP-A: source img with mask painted on top
* EXP-B: source img clean + source img with mask painted on top
* EXP-C: source img clean mask + source img with mask + dest img