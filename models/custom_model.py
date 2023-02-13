import einops
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

max_caption_length = 72

WORKSPACE_PATH = os.getcwd()
VOCAB_PATH = os.path.join(WORKSPACE_PATH, "tokenizer_vocab.pkl")
OUTPUT_LAYER_PATH = os.path.join(WORKSPACE_PATH, "output_layer_bias.pkl")

# decoder model
class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()

        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=max_length, output_dim=depth
        )

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=depth, mask_zero=True
        )

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.token_embedding(seq)  # (batch, seq, depth)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)

        return self.add([seq, x])

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x, use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, y, **kwargs):
        attn, attention_scores = self.mha(
            query=x, value=y, return_attention_scores=True
        )

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()

        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=2 * units, activation="relu"),
                tf.keras.layers.Dense(units=units),
                tf.keras.layers.Dropout(rate=dropout_rate),
            ]
        )

        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=units, dropout=dropout_rate
        )
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        in_seq, out_seq = inputs

        # Text input
        out_seq = self.self_attention(out_seq)

        out_seq = self.cross_attention(out_seq, in_seq)

        self.last_attention_scores = self.cross_attention.last_attention_scores

        out_seq = self.ff(out_seq)

        return out_seq

class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=("", "[UNK]", "[START]"), bias=None, **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.bias = bias
    
    def set_bias(self, bias):
        self.bias = bias

    def call(self, x):
        x = self.dense(x) 
        return x + self.bias

class Captioner(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(
        self,
        tokenizer,
        feature_extractor,
        output_layer,
        num_layers=2,
        units=256,
        max_length=max_caption_length+1, # account for [END] token
        num_heads=2,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.output_layer = output_layer
        self.num_layers = num_layers
        self.units = units
        self.max_length = max_length
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="", vocabulary=tokenizer.get_vocabulary()
        )
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
        )

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(), depth=units, max_length=self.max_length
        )

        self.decoder_layers = [
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for n in range(num_layers)
        ]

    def call(self, inputs):
        image, txt = inputs

        if image.shape[-1] == 3:
            # Apply the feature-extractor, if you get an RGB image.
            image = self.feature_extractor(image)
        # Flatten the feature map
        image = einops.rearrange(image, 'b h w c -> b (h w) c')

        if txt.dtype == tf.string:
            # Apply the tokenizer if you get string inputs.
            txt = tokenizer(txt)
        txt = self.seq_embedding(txt)

        # Look at the image
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))
        txt = self.output_layer(txt)

        return txt

    def simple_gen(self, image, temperature=0.0):
        # generate captions for the model
        initial = self.word_to_index([["[START]"]])  # (batch, sequence)
        img_features = self.feature_extractor(image[tf.newaxis, ...])

        tokens = initial  # (batch, sequence)
        for n in range(self.max_length):
            preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
            preds = preds[:, -1, :]  # (batch, vocab)
            if temperature == 0:
                next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
            else:
                next = tf.random.categorical(
                    preds / temperature, num_samples=1
                )  # (batch, 1)
            tokens = tf.concat([tokens, next], axis=1)  # (batch, sequence)

            if next[0] == self.word_to_index("[END]"):
                break
        words = self.index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=" ")
        return result.numpy().decode()

    def run_and_show_attention(self, image, temperature=0.0):

        def plot_attention_maps(image, str_tokens, attention_map):
            fig = plt.figure(figsize=(20, 9))

            len_result = len(str_tokens)

            titles = []
            for i in range(len_result):
                map = attention_map[i]
                grid_size = max(int(np.ceil(len_result / 2)), 2)
                ax = fig.add_subplot(3, grid_size, i + 1)
                titles.append(ax.set_title(str_tokens[i]))
                img = ax.imshow(image)
                ax.imshow(
                    map,
                    cmap="gray",
                    alpha=0.6,
                    extent=img.get_extent(),
                    clim=[0.0, np.max(map)],
                )
                plt.axis("off")

            plt.tight_layout()

            return fig
        
        result_txt = self.simple_gen(image, temperature)
        str_tokens = result_txt.split()
        str_tokens.append("[END]")

        attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
        attention_maps = tf.concat(attention_maps, axis=0)
        attention_maps = einops.reduce(
            attention_maps,
            "batch heads sequence (height width) -> sequence height width",
            height=7,
            width=7,
            reduction="mean",
        )

        attention_plot = plot_attention_maps(image / 255, str_tokens, attention_maps)
        t = plt.suptitle(f"Predicted Caption: {result_txt}")
        t.set_y(1.05)
        
        return attention_plot