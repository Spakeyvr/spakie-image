import mlx.core as mx
import numpy as np

from spakie.config import ModelConfig
from spakie.data.tokenizer import BOS, PAD
from spakie.diffusion.loss import corrupt, diffusion_loss, sample_t
from spakie.diffusion.sampler import sample
from spakie.model import PixelTransformer
from spakie.render import to_rgba


def tiny_model(n_colors=9):
    cfg = ModelConfig(dim=32, depth=2, heads=2, text_vocab=11, text_len=6, n_colors=n_colors, side=4)
    palette = np.random.default_rng(0).integers(0, 256, (n_colors, 3), dtype=np.uint8)
    return PixelTransformer(cfg, palette), palette


def texts(b, model):
    t = np.full((b, model.cfg.text_len), PAD, dtype=np.int32)
    t[:, 0] = BOS
    t[:, 1:3] = 5
    return mx.array(t)


def test_forward_shape_and_loss():
    model, _ = tiny_model()
    img = mx.random.randint(0, 9, (3, 16))
    logits = model(img, texts(3, model))
    assert logits.shape == (3, 16, 9)
    loss = diffusion_loss(model, img, texts(3, model), sample_t(3, 1e-3), mx.random.uniform(shape=(3, 16)))
    # An untrained model should be near the uniform-guess cost of log2(9) bits per pixel.
    assert 0.5 * np.log2(9) < loss.item() < 2.0 * np.log2(9)


def test_caption_changes_output():
    model, _ = tiny_model()
    img = mx.random.randint(0, 9, (2, 16))
    a = np.array(texts(2, model))
    b = a.copy()
    b[:, 2] = 7
    assert not np.allclose(np.array(model(img, mx.array(a))), np.array(model(img, mx.array(b))), atol=1e-5)


def test_corrupt_masks_expected_fraction():
    img = mx.zeros((4, 4096), dtype=mx.int32)
    t = mx.array([0.1, 0.4, 0.7, 1.0])
    x, masked = corrupt(img, t, mx.random.uniform(shape=img.shape), mask_id=9)
    frac = np.array(masked).mean(axis=1)
    assert np.allclose(frac, np.array(t), atol=0.04)
    assert (np.array(x)[np.array(masked)] == 9).all()


def test_sampler_fills_canvas_and_respects_known_pixels():
    model, _ = tiny_model()
    text, empty = texts(4, model), mx.array(np.tile([BOS] + [PAD] * 5, (4, 1)).astype(np.int32))
    for order in ("confidence", "random"):
        out = np.array(sample(model, text, empty, steps=5, cfg=2.0, order=order))
        assert out.shape == (4, 16) and out.min() >= 0 and out.max() < 9

    init = mx.full((4, 16), 3, dtype=mx.int32)
    known = mx.array(np.tile(np.arange(16) < 8, (4, 1)))
    out = np.array(sample(model, text, empty, steps=5, init=init, known=known))
    assert (out[:, :8] == 3).all() and out.max() < 9


def test_render_transparency():
    palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    rgba = to_rgba(np.array([[0, 1, 1, 0]]), palette, side=2)
    assert rgba.shape == (1, 2, 2, 4)
    assert rgba[0, 0, 0].tolist() == [0, 0, 0, 0] and rgba[0, 0, 1].tolist() == [255, 0, 0, 255]
