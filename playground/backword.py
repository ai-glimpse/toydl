import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full", layout_file="layouts/backword.slides.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import toydl
    import math

    from toydl.core.scalar import Scalar
    from toydl.core.scalar.bp import topological_sort
    from toydl.network.mlp import MLPConfig, MLPBinaryClassifyNetFactory, ActivationType

    return (
        ActivationType,
        MLPBinaryClassifyNetFactory,
        MLPConfig,
        Scalar,
        math,
        topological_sort,
        toydl,
    )


@app.cell
def _(toydl):
    toydl.__version__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Simple function backward""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$ y = f(x_1, x_2) = 3x_1 + 5x_2 $$

    $$ f'(x_1) = 3 $$

    $$ f'(x_2)  = 5 $$
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(Scalar):
    x1 = Scalar(1.0, name="x1")
    x1.requires_grad_()
    x2 = Scalar(2.0, name="x2")
    x2.requires_grad_()

    y = Scalar(3, name="x1_coef") * x1 + Scalar(5, name="x2_coef") * x2
    y.name = "y"

    y
    return x1, x2, y


@app.cell
def _(topological_sort, y):
    topological_sort(y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""注意到，除了我们指定的一些 `Scalar` 之外，这里存在一些存储中间计算结果的 `Scalar`."""
    )
    return


@app.cell
def _(math, x1, x2, y):
    y.backward(d_output=1.0)

    assert math.isclose(x1.derivative, 3.0)
    assert math.isclose(x2.derivative, 5.0)
    return


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(x2):
    x2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MLP backward""")
    return


@app.cell
def _(ActivationType, MLPBinaryClassifyNetFactory, MLPConfig):
    mlp_config = MLPConfig(
        in_size=2,
        out_size=1,
        hidden_layer_size=3,
        hidden_layer_num=1,
        hidden_activation=ActivationType.SIGMOID,
    )
    mlp = MLPBinaryClassifyNetFactory(mlp_config)
    return (mlp,)


@app.cell
def _(mlp):
    for _p in mlp.named_parameters():
        print(_p)
    return


@app.cell
def _(Scalar, mlp):
    x1_1 = Scalar(1.0, name="x1")
    x2_1 = Scalar(2.0, name="x2")
    xs = [x1_1, x2_1]
    y_1 = mlp.forward(xs)
    y_1.name = "y"
    y_1
    return x1_1, x2_1, y_1


@app.cell
def _(y_1):
    y_1.backward(d_output=1.0)
    return


@app.cell
def _(mlp):
    for _p in mlp.named_parameters():
        print(_p)
    return


@app.cell
def _(x1_1):
    x1_1
    return


@app.cell
def _(x2_1):
    x2_1
    return


if __name__ == "__main__":
    app.run()
