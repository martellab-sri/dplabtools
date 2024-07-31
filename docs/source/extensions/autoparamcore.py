# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


from textwrap import dedent
import importlib
import uuid
import functools


class AutoParamCore:
    def __init__(self, params, paths):
        self._params = params
        self._paths = paths
        self._docstring = None
        self._docstring_data = None
        self._run()

    def _run(self):
        docstring_data = self._get_docstring_data(self._params, self._paths)
        self._docstring = str(self._create_docstring(docstring_data))

    @staticmethod
    def _get_docstring_data(params, paths):
        all_docstrings_data = []
        for param, path in zip(params, paths):
            module_path, class_name = AutoParamCore._split_path(path)
            module = importlib.import_module(module_path)
            doc_class = getattr(module, class_name)
            doc_class_params_text = doc_class.__init__.__doc__
            class_params_data = AutoParamCore._get_class_params_data(class_name, doc_class_params_text)
            current_param_data = AutoParamCore._find_param_data(param, class_params_data)
            if current_param_data:
                all_docstrings_data.append(current_param_data)
        return all_docstrings_data

    @staticmethod
    def _split_path(path):
        path_list = path.split(".")
        if len(path_list) == 1:
            raise ValueError("Invalid class import path: %s" % path)
        module_path = ".".join(path_list[:-1])
        class_name = path_list[-1]
        return module_path, class_name

    @staticmethod
    @functools.lru_cache
    def _get_class_params_data(class_name, class_params_text):
        class_params_data = []
        class_params_text = dedent(class_params_text)
        tmp_list = class_params_text.split("----------\n")  # "Parameters"
        if len(tmp_list) != 2:
            raise ValueError("Could not find 'Parameters' section in docstring for class %s" % class_name)

        all_params_text = tmp_list[1]
        all_params_list = all_params_text.split("\n\n")

        for param_text in all_params_list:
            # split single parameter first line into two elements only
            param_lines = param_text.split("\n", 1)
            if len(param_lines) != 2 or not (param_lines[0] and param_lines[1]):
                raise ValueError(
                    "Could not find name/type or description for lines '%s' in class %s" % (param_lines, class_name)
                )
            name_and_type, description = param_lines[0].strip(), param_lines[1].strip()
            if not name_and_type:
                raise ValueError(
                    "Received blank parameter name, too many new lines somewhere after '----------' in %s?" % class_name
                )
            # extract name and type
            nametype_split = name_and_type.split(":", 1)
            if len(nametype_split) != 2 or not (nametype_split[0] and nametype_split[1]):
                raise ValueError("Could not find name or type for line '%s' in class %s" % (name_and_type, class_name))
            var_name, var_type = nametype_split[0].strip(), nametype_split[1].strip()
            # remove line breaks and multi-spaces in description
            description = " ".join(description.replace("\n", " ").split())
            # good to go
            class_params_data.append((var_name, var_type, description))
        return class_params_data

    @staticmethod
    def _find_param_data(param_name, class_params_data):
        found_data = None
        for param_data in class_params_data:
            if param_data[0] == param_name:
                found_data = param_data
                break
        if not found_data:
            print("!!!AUTOPARAM WARNING!!! Parameter or docstring not found: %s" % param_name)
        return found_data

    @staticmethod
    def _create_docstring(docstring_data):
        uid = str(uuid.uuid4()).replace("-", "")
        hidden_class_name = "dplabtoolshiddenclass_" + uid
        docstring = dedent(""".. class:: %s \n    :noindex: \n""" % hidden_class_name)

        for doc_elem in docstring_data:
            param_name = doc_elem[0].strip()
            param_type = doc_elem[1].strip()
            param_text = doc_elem[2].strip()
            docstring += "\n    :param %s %s: %s" % (param_type, param_name, param_text)

        return docstring

    @property
    def docstring(self):
        return self._docstring
