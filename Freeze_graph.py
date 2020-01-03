import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import tensorflow


# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = r'\model'
model_fname = r'E:\Estek-AIProject\INTEL-SOOP\Trainings\classifier_pass-soop_31-12-19-MobileNetV2I1_cornersonly64\mymodelCorners64px_37.h5'
#model_fname = r'E:\Estek-AIProject\INTEL-SOOP\Trainings\classifier_pass-soop_24-12-19-MobileNetV2I2_cornersonly64/classifier_pass-soop_24-12-19-MobileNetV2I2_cornersonly64.h5'
def freeze_graph(graph, session, output, save_pb_dir=save_pb_dir, save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        save_pb_dir = 'E:\Estek-AIProject\INTEL-SOOP\keras\model'
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=False)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)
model.summary()
print('Output is: ' + str(model.outputs))

session = tf.compat.v1.keras.backend.get_session()


INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)


#frozen_graph = freeze_graph(session.graph, session, ['dense_2/Sigmoid'], save_pb_dir=save_pb_dir)
#out.op.name for out in model.output

#graphdef_inf = tf.graph_util.remove_training_nodes(session.graph.as_graph_def())
frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), ['output_node/Sigmoid'])
infer_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen)

dest_folder = r'E:\Estek-AIProject\INTEL-SOOP\Trainings\classifier_pass-soop_31-12-19-MobileNetV2I1_cornersonly64\frozen'
tf.compat.v1.io.write_graph(infer_graph, dest_folder, 'frozen-inference.pb', as_text=False)

print('write success!')
