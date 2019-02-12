import os
import soundfile
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
slim = tf.contrib.slim


class Extractor:
    def __init__(self, checkpoint_path='vggish_model.ckpt', pcm_params_path='vggish_pca_params.npz'):
        checkpoint_path = os.path.join(checkpoint_path)
        pcm_params_path = os.path.join(pcm_params_path)
        # Load model

        vggish_slim.define_vggish_slim(training=False)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(self.sess, checkpoint_path)
        self.features_tensor = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        self.pproc = vggish_postprocess.Postprocessor(pcm_params_path)

    ### Feature extraction.
    def extract_audioset_embedding(self, audio_path):
        """Extract log mel spectrogram features.
        """

        # Arguments & parameters
        sample_rate = vggish_params.SAMPLE_RATE
        
        # Read audio
        (audio, fs) = soundfile.read(audio_path)

        # Extract log mel feature
        logmel = vggish_input.waveform_to_examples(audio, sample_rate)

        # Extract embedding feature
        [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: logmel})

        # PCA
        postprocessed_batch = self.pproc.postprocess(embedding_batch)

        return postprocessed_batch
