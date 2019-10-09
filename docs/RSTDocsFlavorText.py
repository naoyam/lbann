# This file contains the headers and flavor text to appear on the
# organizational top pages of the developer documentation. In both
# dictionaries, the keys are paths relative to "include/lbann". The
# header will be the title for the page. The flavor text will be
# inserted under the title and before a toctree listing contents.

lbann_rst_headers = {
    '.' : 'LBANN API',
    'callbacks' : 'Callback Interface',
    'data_readers' : 'Data Readers Interface',
    'data_store' : 'Data Store Interface',
    'execution_contexts' : 'Execution Context Interface',
    'layers' : 'Layer Interface',
    'layers/activations' : 'Activation Layers',
    'layers/image' : 'Image Layers',
    'layers/io' : 'I/O Layers',
    'layers/learning' : 'Learning Layers',
    'layers/math' : 'Math Layers',
    'layers/misc' : 'Miscellaneous Layers',
    'layers/regularizers' : 'Regularization Layers',
    'layers/transform' : 'Transform Layers',
    'io': 'I/O Utilities',
    'io/data_buffers': 'Data Buffers for Data Ingestion',
    'metrics' : 'Metrics Interface',
    'models' : 'Models Interface',
    'objective_functions' : 'Objective Function Interface',
    'objective_functions/weight_regularization' : 'Objective Functions for Weight Regularization',
    'optimizers' : 'Optimizer Interface',
    'proto' : 'Protobuf and Front-End Utilities',
    'trainers' : 'Trainer Interface',
    'training_algorithms' : 'Training Algorithm Interface',
    'transforms' : 'Transform Interface',
    'utils' : 'General Utilities',
    'utils/threads' : 'Multithreading Utilities',
    'weights' : 'Weights Interface'
}

lbann_rst_flavor_text = {
    '.' : '''
Welcome to the LBANN developers' documentation. The documentation is
laid out following a similar structure to the source code to aid in
navigation.
    ''',

    'callbacks' : '''
Callbacks give users information about their model as it is trained.
Users can select which callbacks to use during training in their
model prototext file.''',

    'data_readers' : '''
Data readers provide a mechanism for ingesting data into LBANN.  This
is typically where a user may have to interact with the LBANN source
code.''',

    'data_store' : '''
The data store provides in-memory caching of the data set and
inter-epoch data shuffling.''',

    'execution_contexts' : '''
When a model is attached to a trainer, the execution context of the
training algorithm is stored in an `execution_context` (or sub-class)
object per execution mode.  Thus there is one execution context per
model and mode that contains all of the state with respect to the
training algorithm being applied to the model.

For example it tracks the current:

* step
* execution mode
* epoch
* and a pointer back to the trainer.
''',

    'layers' : '''
LBANN models are defined in model prototext files. The bulk of these
defintions will be the series of layers which make up the model
itself. LBANN layers all inherit from the common base
:code:`lbann::Layer`. The concrete layers belong to one of several
categories.''',

    'io': '''
Classes for persisting the state of LBANN (checkpoint and restart),
 file I/O and data buffers.''',

    'io/data_buffers': '''
The data buffer classes describe how data is distributed across the
input layer. Note that this part of the class hierarchy is scheduled
to be deprecated and folded into the existing input layer class.''',

    'metrics' : '''
A metric function can be used to evaluate the performance of a model
without affecting the training process. Users define the metric with
which to test their model in their model prototext file.
The available metric functions in LBANN are found below.''',

    'models' : '''
A model is a collection of layers that are composed into a
computational graph. The model also holds the weight matrices for each
learning layer. During training the weight matrices are the free
parameters. For a trained network during inference the weight matrics
are preloaded from saved matrices. The model also contains the
objective function and optimizer classes for the weights.''',

    'objective_functions' : '''
An objective function is the measure that training attempts to optimize.
Objective functions are defined in a user's model defintion prototext
file. Available objective functions can be found below.''',

    'objective_functions/weight_regularization' : '''
TODO:Something about objective_functions/weight_regularization''',

    'optimizers' : '''
Optimizer algorithms attempt to optimize model weights. Optimizers
are selected when invoking LBANN via a command line argument
(:code:`--optimizer=<path_top_opt_proto>`). Available optimizers
are found below.''',

    'proto' : '''
LBANN uses the Tensorflow protobuf format for specifying the
architecture of neural networks, data readers, and optimizers. It
serves as the "assembly language" interface to the toolkit. The
python front end of LBANN will emit a network description in the
protobuf format that is ingested at runtime.''',

    'trainers' : '''
A trainer is a collection of compute resources and defines an explicit
communication domain.  It manages the execution for both the training
and inference of a trained model.  Once constructed, a trainer owns an
`lbann_comm` object that defines both intra- and inter-trainer
communication domains.  Additionally, a trainer will contain an I/O
thread pool that is used to fetch and preprocess data that will be
provided to the trainer's models.

A trainer owns:

* `lbann_comm` object,
* I/O thread pool,
* One or more models, and
* Execution context for each model.

In the future, it will also contain the data readers.
''',

    'training_algorithms' : '''
The training algorithm defines the optimization that is to be
applied to the model(s) being trained.  Additionally, it can
specify how to evaluate the model.
''',

    'utils' : 'Utility classes and functions.',

    'utils/threads' : 'TODO: Something about utils/threads',

    'weights' : '''
The weight class is the representation of the trainable parameters in
the neural network.  Learning layers each have an independent weight
class.  During stochastic gradient descent training the weight
matrices are updated after each forward and backward propagation step.'''
}
