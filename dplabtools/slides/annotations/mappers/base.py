# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for mapping annotation data.

Mapped data should have standardized names (keys and values) for common data elements as well as all annotation
specific issues resolved (e.g. handling different polygon types) using internal static methods.
"""


class BaseMapper:
    """Base class for annotation mappers."""

    label_name = "user_label"
    points_name = "points"

    def __init__(
        self,
        *,
        annotations,
        get_label_fn,
        builtin_key_mappings,
        builtin_value_mappings,
        custom_key_mappings={},
        custom_value_mappings={},
        extra_keys=[],
    ):
        self._annotations = annotations
        self._get_label_fn = get_label_fn
        self._builtin_key_mappings = builtin_key_mappings
        self._builtin_value_mappings = builtin_value_mappings
        self._custom_key_mappings = custom_key_mappings
        self._custom_value_mappings = custom_value_mappings
        self._extra_keys = extra_keys
        self._mapped_annotations = []
        self._map_data()

    def _map_data(self):
        for annotation in self._annotations:
            new_data = {}
            # add user label
            BaseMapper._add_keys(annotation, [self.label_name])
            # add extra fields/keys
            BaseMapper._add_keys(annotation, self._extra_keys)
            for k, v in annotation.items():
                # map values
                if k in self._custom_value_mappings:
                    custom_function = self._custom_value_mappings[k]
                    mapped_value = custom_function(**annotation)
                elif k in self._builtin_value_mappings:
                    method_name = self._builtin_value_mappings[k]
                    builtin_method = self._get_mapper_method(method_name)
                    mapped_value = builtin_method(**annotation)
                else:
                    mapped_value = v
                # map keys
                if k in self._custom_key_mappings:
                    mapped_key = self._custom_key_mappings[k]
                elif k in self._builtin_key_mappings:
                    mapped_key = self._builtin_key_mappings[k]
                else:
                    mapped_key = k
                # assign new key/value pair
                new_data[mapped_key] = mapped_value
            # after finishing processing one annotation update its label
            new_data[self.label_name] = self._get_label_fn(**new_data)
            self._mapped_annotations.append(new_data)

    @staticmethod
    def _add_keys(data_dict, extra_keys):
        for key in extra_keys:
            if key in data_dict:
                raise ValueError("Key '%s' already exists in annotation" % key)
            data_dict.update({key: ""})

    @classmethod
    def _get_mapper_method(cls, method_name):
        return getattr(cls, method_name)

    @property
    def annotations(self):
        """Return mapped annotations as list."""
        return self._mapped_annotations
