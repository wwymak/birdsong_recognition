import Algorithmia
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub

client = Algorithmia.client()


# Configure Tensorflow to only use up to 30% of the GPU.
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3432)])

bird_mapping_idx_to_name = {0: 'barn_swallow', 1: 'black-headed_gull', 2: 'black_woodpecker', 3: 'carrion_crow',
                            4: 'coal_tit',
                            5: 'common_blackbird', 6: 'common_chaffinch', 7: 'common_chiffchaff', 8: 'common_cuckoo',
                            9: 'common_house_martin', 10: 'common_linnet', 11: 'common_moorhen',
                            12: 'common_nightingale',
                            13: 'common_pheasant', 14: 'common_redpoll', 15: 'common_redshank', 16: 'common_redstart',
                            17: 'common_reed_bunting', 18: 'common_snipe', 19: 'common_starling', 20: 'common_swift',
                            21: 'common_whitethroat', 22: 'common_wood_pigeon', 23: 'corn_bunting', 24: 'dunlin',
                            25: 'dunnock',
                            26: 'eurasian_blackcap', 27: 'eurasian_blue_tit', 28: 'eurasian_bullfinch',
                            29: 'eurasian_collared_dove', 30: 'eurasian_coot', 31: 'eurasian_golden_oriole',
                            32: 'eurasian_jay',
                            33: 'eurasian_magpie', 34: 'eurasian_nuthatch', 35: 'eurasian_oystercatcher',
                            36: 'eurasian_reed_warbler', 37: 'eurasian_skylark', 38: 'eurasian_tree_sparrow',
                            39: 'eurasian_treecreeper', 40: 'eurasian_wren', 41: 'eurasian_wryneck',
                            42: 'european_bee-eater',
                            43: 'european_golden_plover', 44: 'european_goldfinch', 45: 'european_green_woodpecker',
                            46: 'european_greenfinch', 47: 'european_herring_gull', 48: 'european_honey_buzzard',
                            49: 'european_nightjar', 50: 'european_robin', 51: 'european_turtle_dove',
                            52: 'garden_warbler',
                            53: 'goldcrest', 54: 'great_spotted_woodpecker', 55: 'great_tit', 56: 'grey_partridge',
                            57: 'house_sparrow', 58: 'lesser_whitethroat', 59: 'long-tailed_tit', 60: 'marsh_tit',
                            61: 'marsh_warbler', 62: 'meadow_pipit', 63: 'northern_lapwing', 64: 'northern_raven',
                            65: 'red-throated_loon', 66: 'red_crossbill', 67: 'redwing', 68: 'river_warbler',
                            69: 'rock_dove',
                            70: 'rook', 71: 'sedge_warbler', 72: 'song_thrush', 73: 'spotted_flycatcher',
                            74: 'stock_dove',
                            75: 'tawny_owl', 76: 'tree_pipit', 77: 'western_yellow_wagtail', 78: 'willow_ptarmigan',
                            79: 'willow_tit', 80: 'willow_warbler', 81: 'wood_sandpiper', 82: 'wood_warbler',
                            83: 'yellowhammer'}

bird_mapping_name_to_idx = {v: k for k, v in bird_mapping_idx_to_name.items()}

class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


def load_model():
    """Load model from data collection."""
    file_uri = "data://shadyvale/birdsong_classifier_models/model_all_birds_v1.h5"
    # Retrieve file name from data collections.
    saved_model_path = client.file(file_uri).getFile().name
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'ReduceMeanLayer': ReduceMeanLayer,
                                                                                  'KerasLayer': hub.KerasLayer})

    return model


# Function to load model gets called one time
classifier = load_model()


@tf.function
def load_wav_16k_mono(wav_binary):
    wav, sample_rate = tf.audio.decode_wav(
        wav_binary,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def predict(input):
    wav_data = load_wav_16k_mono(input)

    results = classifier(wav_data)
    top_probs, top_idxs = tf.math.top_k(results, k=5)
    _, most_likely_idx = tf.math.top_k(results, k=1)
    top_labels = [bird_mapping_idx_to_name[idx] for idx in top_idxs.numpy()]
    return top_labels


def apply(input):
    """Pass in a csv image file and output prediction."""
    output = predict(input)
    print(output)
    return output


if __name__ == '__main__':
    with open('/media/wwymak/Storage2/birdsong_dataset/xeno_canto_eu_cleaned/willow_tit/402797.wav', 'rb') as f:
        file_contents = f.read()
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        res = apply(file_contents)
        print('here')