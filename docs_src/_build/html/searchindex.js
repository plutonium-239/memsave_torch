Search.setIndex({"docnames": ["api/index", "api/nn", "api/util/collect_results", "api/util/estimate", "api/util/index", "api/util/measurements", "api/util/models", "basic_usage", "index"], "filenames": ["api\\index.rst", "api\\nn.rst", "api\\util\\collect_results.rst", "api\\util\\estimate.rst", "api\\util\\index.rst", "api\\util\\measurements.rst", "api\\util\\models.rst", "basic_usage.rst", "index.rst"], "titles": ["API Reference", "nn", "ResultsCollector", "estimate.py", "util", "&lt;no title&gt;", "Models", "Installation / Quickstart", "Welcome to MemSave PyTorch\u2019s documentation!"], "terms": {"util": [0, 2, 3, 6, 8], "packag": 0, "The": [0, 1, 2, 3, 6, 7], "consist": 0, "two": [0, 6], "main": 0, "thi": [0, 1, 2, 3, 4, 6], "tri": 0, "mirror": 0, "torch": [0, 1, 6, 7], "offer": 0, "follow": [0, 1, 3], "layer": [0, 3], "readili": 0, "replac": [0, 1], "call": [0, 2, 6], "convert_to_memory_sav": [0, 1, 7], "function": [0, 2, 3, 6, 7], "modul": [1, 3, 6, 7, 8], "contain": 1, "member": 1, "memsaveconv2d": 1, "memsavelinear": 1, "memsaverelu": 1, "memsavemaxpool2d": 1, "memsavebatchnorm2d": 1, "memsavelayernorm": 1, "implement": [], "memori": [1, 2, 3, 4, 6, 7], "save": [1, 2, 3, 7], "neural": 3, "network": [1, 3], "current": [], "linear": [1, 2, 3], "conv2d": 1, "batchnorm2d": 1, "memsave_torch": [1, 4, 6, 7], "model": [1, 2, 3, 4, 7], "true": [1, 3], "relu": [1, 6], "maxpool2d": 1, "layernorm": 1, "verbos": 1, "fals": [1, 3], "clone_param": 1, "convert": [1, 3], "given": [1, 2, 6], "": [1, 6], "memsav": 1, "version": 1, "option": [1, 2, 3], "choos": 1, "which": [1, 2, 3, 6], "type": [1, 2, 3, 6], "should": 1, "us": [1, 2], "when": [1, 3], "you": 1, "plan": 1, "both": 1, "simultan": 1, "otherwis": [1, 6], "grad": [1, 3], "accumul": 1, "one": [1, 3, 6], "wll": 1, "affect": 1, "other": 1, "sinc": 1, "weight": [1, 3], "ar": [1, 2], "same": [1, 3], "tensor": [1, 3], "object": [1, 6], "For": [1, 6], "an": [1, 3], "exampl": [1, 6], "see": [1, 3], "test": 1, "test_lay": 1, "py": [1, 4], "paramet": [1, 2, 3, 6], "input": [1, 3], "bool": [1, 3], "whether": [1, 3], "print": [1, 2, 3], "were": 1, "clone": 1, "directli": 1, "return": [1, 2, 3, 6], "memsavemodel": 1, "class": [1, 2], "in_channel": 1, "int": [1, 2], "out_channel": 1, "kernel_s": 1, "stride": 1, "1": [1, 6], "pad": 1, "0": 1, "dilat": 1, "group": 1, "bia": 1, "padding_mod": 1, "str": [1, 2, 3, 6], "zero": 1, "devic": [1, 2, 3], "none": [1, 2, 3], "dtype": 1, "init": 1, "param": 1, "forward": [1, 3, 4], "pass": [1, 3, 4, 7], "b": 1, "c_in": 1, "h": 1, "w": 1, "output": [1, 2, 6], "c_out": 1, "h_out": 1, "w_out": 1, "classmethod": 1, "from_nn_conv2d": 1, "obj": 1, "in_featur": 1, "out_featur": 1, "x": [1, 3], "f_in": 1, "f_out": 1, "from_nn_linear": 1, "from_nn_relu": 1, "return_indic": 1, "ceil_mod": 1, "from_nn_maxpool2d": 1, "num_featur": 1, "ep": 1, "1e": 1, "05": 1, "momentum": 1, "affin": 1, "track_running_stat": 1, "c": 1, "from_nn_batchnorm2d": 1, "bn2d": 1, "normalized_shap": 1, "elementwise_affin": 1, "introduc": 1, "v2": 1, "from_nn_layernorm": 1, "ln": 1, "resultscollector": 4, "handl": 2, "collect": [2, 4], "organ": 2, "result": [2, 3], "from": [2, 6, 7], "temp": 2, "file": [2, 6], "gener": 2, "estim": [2, 4, 6], "script": [2, 3], "make": [2, 6], "datafram": 2, "all": [2, 3], "case": [2, 3], "batch_siz": 2, "input_channel": 2, "input_hw": 2, "num_class": 2, "architectur": [2, 3], "vjp_improv": 2, "list": [2, 3, 6], "float": 2, "results_dir": [2, 3], "callabl": [2, 3], "bound": 2, "method": 2, "tqdm": 2, "write": 2, "std": 2, "read": 2, "directori": 2, "initi": [2, 6], "collector": 2, "befor": 2, "run": [2, 3, 6], "conv": [2, 3], "union": 2, "base": 2, "dir": 2, "i": [2, 3, 4, 7], "e": 2, "caus": [2, 3], "problem": 2, "context": 2, "clear_fil": 2, "clear": [2, 6], "after": [2, 3, 6], "stat": 2, "time": [2, 3, 4], "collect_from_fil": 2, "To": 2, "have": [2, 3, 6], "finish": 2, "name": [2, 6], "rais": 2, "assertionerror": 2, "If": 2, "ha": [2, 3], "more": [2, 6], "less": 2, "line": 2, "than": 2, "expect": 2, "number": 2, "valueerror": 2, "unallow": 2, "text": 2, "onli": [2, 3], "allow": 2, "been": 2, "csv": 2, "hyperparam_str": 2, "arg": [2, 3], "simplenamespac": 2, "format": 2, "hyperparam": 2, "string": [2, 6], "make_case_str": 2, "obsrev": 3, "peak": 3, "total": 3, "taken": 3, "till": 3, "backward": [3, 4], "possibl": 3, "speed": 3, "up": 3, "random": 3, "vjp": 3, "convolut": 3, "we": 3, "take": 3, "cnn": 3, "answer": 3, "question": 3, "q1": 3, "what": 3, "rel": 3, "consum": [3, 4], "q2": 3, "assum": 3, "achiev": 3, "would": 3, "optim": 3, "step": 3, "q3": 3, "term": 3, "consumpt": [3, 4], "estimate_mem_sav": 3, "model_fn": 3, "loss_fn": 3, "y": 3, "target": 3, "dict": [3, 6], "dev": 3, "return_v": 3, "set": 3, "loss": 3, "label": 3, "detect": 3, "comput": 3, "indic": 3, "valu": [3, 6], "default": 3, "requir": [3, 6], "estimate_speedup": 3, "train": 3, "parse_cas": 3, "small": 3, "helper": 3, "kw": 3, "argument": [3, 6], "measur": [3, 4, 6], "dictionari": [3, 6], "kei": [3, 6], "allowed_cas": 3, "present": 3, "dont": 3, "start": 3, "no_": 3, "skip_case_check": 3, "namespac": 3, "decid": 3, "skip": 3, "grad_norm_": 3, "doe": [3, 6], "ani": 3, "normal": 3, "argpars": 3, "A": 4, "dure": 4, "variou": [4, 6], "collect_result": 2, "pip": 7, "git": 7, "http": 7, "github": 7, "com": 7, "plutonium": 7, "239": 7, "nn": [7, 8], "handi": 7, "tool": 7, "counterpart": 7, "import": 7, "my_torch_model": 7, "memsave_torch_model": 7, "instal": 8, "api": 8, "index": 8, "search": 8, "page": 8, "conv1d": 1, "dropout": 1, "experi": [2, 3, 4, 6], "submodul": 4, "part": 4, "natur": 4, "being": [4, 6], "experiment": 4, "subject": 4, "mainten": 4, "defin": 6, "map": 6, "wa": 6, "necessari": 6, "isol": 6, "separ": 6, "runtim": 6, "everi": 6, "singl": 6, "cuda": 6, "unless": 6, "absolut": 6, "even": 6, "empty_cach": 6, "veri": 6, "difficult": 6, "add": 6, "conv_model_fn": 6, "prefix_in_pair": 6, "prefix": 6, "each": 6, "entri": 6, "resnet101": 6, "convnext": 6, "memsave_": 6, "memsave_resnet101": 6, "memsave_convnext": 6, "ad": 6, "descript": 6, "iter": 6, "item": 6, "pair": 6, "alexnet": 6, "convnext_bas": 6, "deeplabv3_resnet101": 6, "deepmodel": 6, "_conv_model1": 6, "deeprelumodel": 6, "_convrelu_model1": 6, "deeprelupoolmodel": 6, "_convrelupool_model1": 6, "efficientnet_v2_l": 6, "fasterrcnn_resnet50_fpn_v2": 6, "fcn_resnet101": 6, "gpt2": 6, "lambda": 6, "memsave_alexnet": 6, "memsave_convnext_bas": 6, "memsave_deeplabv3_resnet101": 6, "memsave_deepmodel": 6, "_conv_model2": 6, "memsave_deeprelumodel": 6, "memsave_deeprelupoolmodel": 6, "memsave_efficientnet_v2_l": 6, "memsave_fasterrcnn_resnet50_fpn_v2": 6, "memsave_fcn_resnet101": 6, "memsave_gpt2": 6, "memsave_mobilenet_v3_larg": 6, "memsave_resnet101_conv": 6, "bn": 6, "memsave_resnet101_conv_ful": 6, "memsave_resnet18": 6, "memsave_resnext101_64x4d": 6, "memsave_retinanet_resnet50_fpn_v2": 6, "memsave_ssdlite320_mobilenet_v3_larg": 6, "memsave_vgg16": 6, "mobilenet_v3_larg": 6, "resnet18": 6, "resnext101_64x4d": 6, "retinanet_resnet50_fpn_v2": 6, "ssdlite320_mobilenet_v3_larg": 6, "vgg16": 6, "new": 6, "empti": 6, "via": 6, "d": 6, "k": 6, "v": 6, "kwarg": 6, "keyword": 6, "2": 6, "quickstart": 8, "refer": 8}, "objects": {"experiments": [[4, 0, 0, "-", "util"]], "experiments.util": [[2, 0, 0, "-", "collect_results"], [3, 0, 0, "-", "estimate"], [6, 0, 0, "-", "models"]], "experiments.util.collect_results": [[2, 1, 1, "", "ResultsCollector"], [2, 3, 1, "", "hyperparam_str"], [2, 3, 1, "", "make_case_str"]], "experiments.util.collect_results.ResultsCollector": [[2, 2, 1, "", "clear_file"], [2, 2, 1, "", "collect_from_file"], [2, 2, 1, "", "finish"]], "experiments.util.estimate": [[3, 3, 1, "", "estimate_mem_savings"], [3, 3, 1, "", "estimate_speedup"], [3, 3, 1, "", "parse_case"], [3, 3, 1, "", "skip_case_check"]], "experiments.util.models": [[6, 4, 1, "", "conv_model_fns"], [6, 3, 1, "", "prefix_in_pairs"]], "": [[0, 0, 0, "-", "memsave_torch"]], "memsave_torch": [[1, 0, 0, "-", "nn"]], "memsave_torch.nn": [[1, 1, 1, "", "MemSaveBatchNorm2d"], [1, 1, 1, "", "MemSaveConv2d"], [1, 1, 1, "", "MemSaveLayerNorm"], [1, 1, 1, "", "MemSaveLinear"], [1, 1, 1, "", "MemSaveMaxPool2d"], [1, 1, 1, "", "MemSaveReLU"], [1, 3, 1, "", "convert_to_memory_saving"]], "memsave_torch.nn.MemSaveBatchNorm2d": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_BatchNorm2d"]], "memsave_torch.nn.MemSaveConv2d": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_Conv2d"]], "memsave_torch.nn.MemSaveLayerNorm": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_LayerNorm"]], "memsave_torch.nn.MemSaveLinear": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_Linear"]], "memsave_torch.nn.MemSaveMaxPool2d": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_MaxPool2d"]], "memsave_torch.nn.MemSaveReLU": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "from_nn_ReLU"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function", "4": "py:data"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"], "4": ["py", "data", "Python data"]}, "titleterms": {"api": 0, "avail": 0, "modul": 0, "memsave_torch": 0, "nn": [0, 1], "learnabl": 1, "layer": [1, 7], "activ": 1, "pool": 1, "normal": 1, "collect_result": [], "py": 3, "estim": 3, "util": 4, "class": 4, "script": 4, "instal": 7, "replac": 7, "all": 7, "valid": 7, "memsav": [7, 8], "welcom": 8, "pytorch": 8, "": 8, "document": 8, "get": 8, "start": 8, "indic": 8, "tabl": 8, "refer": 0, "resultscollector": 2, "model": 6, "quickstart": 7}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 58}, "alltitles": {"API Reference": [[0, "api-reference"]], "Available Modules": [[0, null]], "memsave_torch.nn": [[0, "memsave-torch-nn"]], "nn": [[1, "nn"]], "Learnable Layers": [[1, "learnable-layers"]], "Activations and Pooling Layers": [[1, "activations-and-pooling-layers"]], "Normalization Layers": [[1, "normalization-layers"]], "ResultsCollector": [[2, "resultscollector"]], "estimate.py": [[3, "estimate-py"]], "util": [[4, "util"]], "Utility classes and scripts": [[4, null]], "Models": [[6, "models"]], "Installation / Quickstart": [[7, "installation-quickstart"]], "Replace all (valid) layers with MemSave layers": [[7, "replace-all-valid-layers-with-memsave-layers"]], "Welcome to MemSave PyTorch\u2019s documentation!": [[8, "welcome-to-memsave-pytorch-s-documentation"]], "Getting started": [[8, null]], "Indices and tables": [[8, "indices-and-tables"]]}, "indexentries": {"memsave_torch": [[0, "module-memsave_torch"]], "module": [[0, "module-memsave_torch"], [1, "module-memsave_torch.nn"], [2, "module-experiments.util.collect_results"], [3, "module-experiments.util.estimate"], [4, "module-experiments.util"], [6, "module-experiments.util.models"]], "memsavebatchnorm2d (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveBatchNorm2d"]], "memsaveconv2d (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveConv2d"]], "memsavelayernorm (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveLayerNorm"]], "memsavelinear (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveLinear"]], "memsavemaxpool2d (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveMaxPool2d"]], "memsaverelu (class in memsave_torch.nn)": [[1, "memsave_torch.nn.MemSaveReLU"]], "convert_to_memory_saving() (in module memsave_torch.nn)": [[1, "memsave_torch.nn.convert_to_memory_saving"]], "forward() (memsave_torch.nn.memsavebatchnorm2d method)": [[1, "memsave_torch.nn.MemSaveBatchNorm2d.forward"]], "forward() (memsave_torch.nn.memsaveconv2d method)": [[1, "memsave_torch.nn.MemSaveConv2d.forward"]], "forward() (memsave_torch.nn.memsavelayernorm method)": [[1, "memsave_torch.nn.MemSaveLayerNorm.forward"]], "forward() (memsave_torch.nn.memsavelinear method)": [[1, "memsave_torch.nn.MemSaveLinear.forward"]], "forward() (memsave_torch.nn.memsavemaxpool2d method)": [[1, "memsave_torch.nn.MemSaveMaxPool2d.forward"]], "forward() (memsave_torch.nn.memsaverelu method)": [[1, "memsave_torch.nn.MemSaveReLU.forward"]], "from_nn_batchnorm2d() (memsave_torch.nn.memsavebatchnorm2d class method)": [[1, "memsave_torch.nn.MemSaveBatchNorm2d.from_nn_BatchNorm2d"]], "from_nn_conv2d() (memsave_torch.nn.memsaveconv2d class method)": [[1, "memsave_torch.nn.MemSaveConv2d.from_nn_Conv2d"]], "from_nn_layernorm() (memsave_torch.nn.memsavelayernorm class method)": [[1, "memsave_torch.nn.MemSaveLayerNorm.from_nn_LayerNorm"]], "from_nn_linear() (memsave_torch.nn.memsavelinear class method)": [[1, "memsave_torch.nn.MemSaveLinear.from_nn_Linear"]], "from_nn_maxpool2d() (memsave_torch.nn.memsavemaxpool2d class method)": [[1, "memsave_torch.nn.MemSaveMaxPool2d.from_nn_MaxPool2d"]], "from_nn_relu() (memsave_torch.nn.memsaverelu class method)": [[1, "memsave_torch.nn.MemSaveReLU.from_nn_ReLU"]], "memsave_torch.nn": [[1, "module-memsave_torch.nn"]], "resultscollector (class in experiments.util.collect_results)": [[2, "experiments.util.collect_results.ResultsCollector"]], "clear_file() (experiments.util.collect_results.resultscollector method)": [[2, "experiments.util.collect_results.ResultsCollector.clear_file"]], "collect_from_file() (experiments.util.collect_results.resultscollector method)": [[2, "experiments.util.collect_results.ResultsCollector.collect_from_file"]], "experiments.util.collect_results": [[2, "module-experiments.util.collect_results"]], "finish() (experiments.util.collect_results.resultscollector method)": [[2, "experiments.util.collect_results.ResultsCollector.finish"]], "hyperparam_str() (in module experiments.util.collect_results)": [[2, "experiments.util.collect_results.hyperparam_str"]], "make_case_str() (in module experiments.util.collect_results)": [[2, "experiments.util.collect_results.make_case_str"]], "estimate_mem_savings() (in module experiments.util.estimate)": [[3, "experiments.util.estimate.estimate_mem_savings"]], "estimate_speedup() (in module experiments.util.estimate)": [[3, "experiments.util.estimate.estimate_speedup"]], "experiments.util.estimate": [[3, "module-experiments.util.estimate"]], "parse_case() (in module experiments.util.estimate)": [[3, "experiments.util.estimate.parse_case"]], "skip_case_check() (in module experiments.util.estimate)": [[3, "experiments.util.estimate.skip_case_check"]], "experiments.util": [[4, "module-experiments.util"]], "conv_model_fns (in module experiments.util.models)": [[6, "experiments.util.models.conv_model_fns"]], "experiments.util.models": [[6, "module-experiments.util.models"]], "prefix_in_pairs() (in module experiments.util.models)": [[6, "experiments.util.models.prefix_in_pairs"]]}})