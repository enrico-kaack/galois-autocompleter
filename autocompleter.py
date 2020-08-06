import json
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tqdm import tqdm
import model, sample, encoder
import logging


class Autocompleter():
    def __init__(self):
        pass

    """
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    def load_model(self, model_name='model', seed=99, nsamples=5, batch_size=5,
                    length=8, temperature=0, top_k=10, top_p=.85, models_dir=''):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))


        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0


        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.models_dir = models_dir

        self.enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0,
                        allow_soft_placement=True, gpu_options=gpu_options)

        self.session =  tf.Session(graph=tf.Graph(), config=config)
        self.session.__enter__()
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # p = tf.random.uniform((1,), minval=.68, maxval=.98, dtype=tf.dtypes.float32, name='random_p_logits')
        self.output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(self.session, ckpt)

    def predict(self, text):
        context_tokens = self.enc.encode(text)
        generated = 0
        predictions = []

        for _ in range(self.nsamples // self.batch_size):

            feed_dict = {self.context: [context_tokens for _ in range(self.batch_size)]}
            logging.debug(f"FEED DICT {feed_dict}")
            out = self.session.run(self.output, feed_dict=feed_dict)[:, len(context_tokens):]
            logging.debug(f"OUT {out}")
            for i in range(self.batch_size):
                generated += 1
                text = self.enc.decode_to_array(out[i])
                predictions.append(text)
        logging.debug(f"PREDICTIONS {predictions}")
    
    def predict_for_tokens(self, context_tokens):
        generated = 0
        predictions = []

        for _ in range(self.nsamples // self.batch_size):
            feed_dict = {self.context: [context_tokens for _ in range(self.batch_size)]}
            out = self.session.run(self.output, feed_dict=feed_dict)[:, len(context_tokens):]
            for i in range(self.batch_size):
                generated += 1
                #text = self.enc.decode_to_array(out[i])
                predictions.append(out[i])
        return predictions

    """
        calcs the score on 0..1 for the token depending on the position of the target in recommendations.
        1 being the best (first recpommendation item is target)
        0 is worst (target not in recommendations)
    """
    def calc_score_for_prediction(self, target, recommendations):
        if target in recommendations:
            # score from 0..1 with 1 being the highest score (first element in the recommendations)
            return (len(recommendations) - recommendations.index(target)) / len(recommendations)
        else:
            return 0


    def predict_for_file_iteratively(self, file_path):
        with open(file_path) as f:
            text = f.read()
            tokens = self.enc.encode(text)
            predictions = []

            for index, token in enumerate(tqdm(tokens)):
                #dont predict for the first token
                if index > 2:
                    target_token = token
                    context_tokens = tokens[:index-1]
                    prediction = self.predict_for_tokens(context_tokens)
                    prediction_decoded = [{"dec": self.enc.decode(raw_prediction), "raw": raw_prediction.tolist()}  for raw_prediction in prediction]
                    predictions.append({"target": {"raw": target_token, "dec": self.enc.decode([target_token])}, "predicted_tokens": prediction_decoded, "score": self.calc_score_for_prediction(target_token, prediction)})
                else:
                    predictions.append({"target": {"raw": token, "dec": self.enc.decode([token])}, "predicted_tokens": [], "score": None})
            return predictions

if __name__ == "__main__":
    logging.basicConfig(level="INFO")


    autocompleter = Autocompleter()
    autocompleter.load_model(model_name="345M", models_dir="models", length=1)

    predictions = autocompleter.predict_for_file_iteratively("/home/enrico/UniProjects/Masterarbeit/analysisPlattform/test_programs/condition_without_method_call.py")
    #predictions = autocompleter.predict_for_file_iteratively("/home/enrico/UniProjects/Masterarbeit/galois-autocompleter/hello.py")
    logging.debug(predictions)
    with open("output.json", "w") as out:
        out.write(json.dumps(predictions))
        