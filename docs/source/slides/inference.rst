=========
Inference
=========

``dplabtools`` offers a simple class for performing inference on WSIs using trained PyTorch models.
The inference process is integrated with other classes included in the package: patch location computing,
patch extraction and heatmap generation.

``WSIInference`` class features:

* Support for segmentation and classification models.
* Support for single and multi resolution patches using dedicated :doc:`/slides/datasets`.
* Support for multi class model output.
* Configurable GPU/CPU processing.
* Inference output integrated with the :doc:`/slides/heatmap` class.
* Ability to save results as images, also with resolution information embedded.
* Parallelization based on PyTorch DataLoaders.


Basic usage
===========

Assuming that the variables ``dataset``, ``model`` and ``classifier`` exist and represent objects:

.. code-block:: python

    from dplabtools.slides.processing import WSIInference

    inference = WSIInference(
        model=model,
        classifier=classifier,
        level_or_minsize=0,
        num_classes=3,
        num_workers=12,
        batch_size=256,
    )
    inference.process_dataset(dataset)


.. _inference-full-example-label:

Full example of inference process
=================================

Full inference process consists of the following steps:

1. WSI mask generation
2. Calculating patch location
3. Creating WSI dataset
4. Initializing PyTorch model and classifier
5. Creating WSIInference object
6. Processing WSI dataset

.. code-block:: python

    from dplabtools.slides.processing import WSITissueMask, WSIDataset, WSIInference
    from dplabtools.slides.patches import WholeImageGridPatches

    wsi_file = "/tmp/wsi1.svs"

    # Step 1
    mask = WSITissueMask(wsi_file=wsi_file, level_or_minsize=2)

    # Step 2
    patches = WholeImageGridPatches(wsi_file=wsi_file, mask_data=mask.array)

    # Step 3
    dataset = WSIDataset(patches=patches)

    # Step 4
    model = get_model()
    classifier = get_classifier()

    # Step 5
    inference = WSIInference(model=model, classifier=classifier, level_or_minsize=0, num_classes=2)

    # Step 6
    inference.process_dataset(dataset)

.. note::

    When processing multiple WSIs as one batch steps 4 and 5 should be performed only once, rather than executed
    separately for each WSI.


Model initialization
--------------------

PyTorch model must be properly initialized before passing it to ``WSIInference``, this could be
accomplished using a simple convenience function ``get_model``:

.. code-block:: python

    import torch

    from mymodels import MyPyTorchModel

    MODEL_PATH = "/tmp/modelfile.pth"

    def get_model():
        model = MyPyTorchModel()
        model.load_state_dict(torch.load(MODEL_PATH))
        ...
        return model

Some PyTorch models must be set in evaluation mode when running the inference, this should be set inside ``get_model``
by calling ``eval()``:


.. code-block:: python

    def get_model():
        model = MyPyTorchModel()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        ...
        return model

When using CUDA processing (``WSIInference`` default mode), the model should be loaded into GPU memory inside
``get_model`` by calling ``cuda()``:

.. code-block:: python

    def get_model():
        model = MyPyTorchModel()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.cuda()
        ...
        return model

When using CPU processing first ``WSIInference`` must be initialized with ``use_cuda=False`` and then model weights must
be loaded with ``map_location="cpu"``:

.. code-block:: python

    def get_model():
        model = MyPyTorchModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        ...
        return model

.. note::

    Error: ``Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`` indicates that
    there is discrepancy between how the model was initialized (CUDA/CPU) inside ``get_model`` and how ``WSIInference``
    was created (``use_cuda=True|False``).


Classifier initialization
-------------------------

Classifier is an optional concept representing additional processing layer for models output. Classifier can be
represented by any callable capable of returning ``torch.Tensor`` output. This includes PyTorch models
as well as native python functions. The most basic classifier would perform `softmax` transformation on the model
output, wrapped in ``get_classifier`` function:

.. code-block:: python

    import torch

    def classifier_fn(result):
        probabilities = torch.nn.functional.softmax(result, dim=1)
        return probabilities

    def get_classifier():
        return classifier_fn

    classifier = get_classifier()

In cases when classifier is not desirable its value should be set to `None`.


Class details
=============

.. autoclass:: dplabtools.slides.processing.WSIInference(...)
   :class-doc-from: init
   :members:
   :inherited-members:

.. seealso::
    :ref:`special-level-or-minsize-label`


Additional configuration
========================

By default the module storing the ``WSIInference`` class sets ``cudnn.benchmark = True``. In cases when this setting is
not desirable its value can be reverted in the following way:

.. code-block:: python

    from dplabtools.slides.processing.inference import cudnn
    cudnn.benchmark = False
