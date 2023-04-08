use dfdx::{prelude::*, optim::{Adam, AdamConfig}};

pub const FEATURES: usize = 14;
pub const NUM_FRAMES: usize = 16;
pub const CHANNELS: usize = FEATURES * NUM_FRAMES;

pub const ACTIONS: usize = 4;
pub const OUTPUTS: usize = ACTIONS + 1; // last output channel is the value head after Avgpool2D

pub type DynImage1<const A: usize> = (Const<A>, usize, usize);
pub type DynImage2<const A: usize> = (usize, Const<A>, usize, usize);

pub type InputShape = DynImage1<CHANNELS>;
pub type FrameShape = DynImage1<FEATURES>;
pub type ActionShape = DynImage1<ACTIONS>;
pub type ValueShape = Rank1<1>;

pub type InputBatchShape = DynImage2<CHANNELS>;
pub type ActionBatchShape = DynImage2<ACTIONS>;
pub type ValueBatchShape = (usize, Const<1>);
pub type BatchShape = (usize,);

type Block<const C: usize, const D: usize, const P: usize> = (
    Residual<(
        Conv2D<C, C, D, 1, P>,
        Bias2D<C>,
        ReLU,
        Conv2D<C, C, D, 1, P>,
        Bias2D<C>,
    )>,
    ReLU,
);

type Downsample<const I: usize, const O: usize> = (Conv2D<I, O, 3, 1, 1>, Bias2D<O>, ReLU);

pub type BigNet = (
    Downsample<CHANNELS, 128>,
    (Repeated<Block<128, 3, 1>, 1>, Downsample<128, 64>),
    (Repeated<Block<64, 5, 2>, 2>, Downsample<64, 32>),
    (
        Repeated<Block<32, 9, 4>, 4>,
        Split2<Conv2D<32, ACTIONS, 7, 1, 3>, (Conv2D<32, 1, 7, 1, 3>, AvgPoolGlobal)>,
    ),
);

pub type SmallNet = Split2<
    (
        Downsample<CHANNELS, 64>,
        Block<64, 5, 2>,
        Conv2D<64, ACTIONS, 5, 1, 2>,
    ),
    (
        Downsample<CHANNELS, 64>,
        Block<64, 5, 2>,
        Conv2D<64, 1, 5, 1, 2>,
        AvgPoolGlobal,
    ),
>;

type UNetBlock<const C1: usize, const C2: usize, M> = Upscale2DResidual<(
    (Conv2D<C1, C2, 3, 2, 1>, Bias2D<C2>, ReLU),
    M,
    (Conv2D<C2, C1, 3, 1, 1>, Bias2D<C1>, ReLU),
)>;

// pub type UNet = Split2<
    // (
        // Downsample<CHANNELS, 64>,
        // UNetBlock<64, 128, UNetBlock<128, 256, (Conv2D<256, 256, 3, 1, 1>, Bias2D<256>, ReLU)>>,
        // Conv2D<64, ACTIONS, 1, 1, 0>,
    // ),
    // (
        // Downsample<CHANNELS, 64>,
        // (Block<64, 3, 1>, Conv2D<64, 128, 3, 2, 1>, Bias2D<128>, ReLU),
        // (Block<128, 3, 1>, Conv2D<128, 256, 3, 2, 1>, Bias2D<256>, ReLU),
        // (AvgPoolGlobal, Linear<256, 1>),
    // ),
// >;

pub type UNet = Split2<
    (
        Downsample<CHANNELS, 64>,
        UNetBlock<64, 128, (Conv2D<128, 128, 3, 1, 1>, Bias2D<128>, ReLU)>,
        Conv2D<64, ACTIONS, 1, 1, 0>,
    ),
    (
        Downsample<CHANNELS, 64>,
        (Block<64, 3, 1>, Conv2D<64, 128, 3, 2, 1>, Bias2D<128>, ReLU),
        (AvgPoolGlobal, Linear<128, 1>),
    ),
>;

pub type TinyNet =
    Split2<Conv2D<CHANNELS, ACTIONS, 3, 1, 1>, (Conv2D<CHANNELS, 1, 3, 1, 1>, AvgPoolGlobal)>;

pub fn test<NN: BuildOnDevice<D, f32>, D: Device<f32>>()
where
    NN::Built: Net<D>,
{
    let dev = D::default();
    let net = dev.build_module::<NN, f32>();
    let input = dev.sample_normal_like(&(Const::<CHANNELS>, 20, 20));
    let (policy, value) = net.forward(input);

    println!("{}", policy.as_vec()[0]);
    println!("{}", value.as_vec()[0]);
    println!("{}", net.num_trainable_params());
}

pub fn test2() {
    type D = Cuda;
    let dev = D::default();
    let mut net = dev.build_module::<UNet, f32>();
    let mut opt = Adam::new(&net, AdamConfig {
        lr: 1e-4,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: None,
    });

    let inp = dev.sample_normal::<Rank4<4, CHANNELS, 10, 10>>();
    let expected = dev.ones::<Rank4<4, ACTIONS, 10, 10>>();

    for _ in 0..100 {
        let (out, _) = net.forward(inp.retaped::<OwnedTape<f32, D>>());
        let loss = mse_loss(out, expected.clone());
        println!("{}", loss.array());
        let _ = opt.update(&mut net, &loss.backward());
    }
}

pub trait Net<D: Device<f32>>:
    'static
    + Module<
        Tensor<InputShape, f32, D>,
        Output = (Tensor<ActionShape, f32, D>, Tensor<ValueShape, f32, D>),
    >
    + ModuleMut<
        Tensor<InputShape, f32, D, OwnedTape<f32, D>>,
        Output = (
            Tensor<ActionShape, f32, D, OwnedTape<f32, D>>,
            Tensor<ValueShape, f32, D, OwnedTape<f32, D>>,
        ),
    >
    + Module<
        Tensor<InputBatchShape, f32, D>,
        Output = (
            Tensor<ActionBatchShape, f32, D>,
            Tensor<ValueBatchShape, f32, D>,
        ),
    >
    + ModuleMut<
        Tensor<InputBatchShape, f32, D, OwnedTape<f32, D>>,
        Output = (
            Tensor<ActionBatchShape, f32, D, OwnedTape<f32, D>>,
            Tensor<ValueBatchShape, f32, D, OwnedTape<f32, D>>,
        ),
    >
    + TensorCollection<f32, D>
{
}

impl<T, D: Device<f32>> Net<D> for T where
    T: 'static
        + Module<
            Tensor<InputShape, f32, D>,
            Output = (Tensor<ActionShape, f32, D>, Tensor<ValueShape, f32, D>),
        >
        + ModuleMut<
            Tensor<InputShape, f32, D, OwnedTape<f32, D>>,
            Output = (
                Tensor<ActionShape, f32, D, OwnedTape<f32, D>>,
                Tensor<ValueShape, f32, D, OwnedTape<f32, D>>,
            ),
        >
        + Module<
            Tensor<InputBatchShape, f32, D>,
            Output = (
                Tensor<ActionBatchShape, f32, D>,
                Tensor<ValueBatchShape, f32, D>,
            ),
        >
        + ModuleMut<
            Tensor<InputBatchShape, f32, D, OwnedTape<f32, D>>,
            Output = (
                Tensor<ActionBatchShape, f32, D, OwnedTape<f32, D>>,
                Tensor<ValueBatchShape, f32, D, OwnedTape<f32, D>>,
            ),
        >
        + TensorCollection<f32, D>
{
}

/// Version of SplitInto the keeps the output tapes of left and right seperate
#[derive(Clone)]
pub struct Split2<L, R> {
    l: L,
    r: R,
}

impl<E: Dtype, D: Device<E>, L: TensorCollection<E, D>, R: TensorCollection<E, D>>
    TensorCollection<E, D> for Split2<L, R>
{
    type To<E2: Dtype, D2: Device<E2>> = Split2<L::To<E2, D2>, R::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("l", |s| &s.l, |s| &mut s.l),
                Self::module("r", |s| &s.r, |s| &mut s.r),
            ),
            |(l, r)| Split2 { l, r },
        )
    }
}

impl<E: Dtype, D: Device<E>, L: BuildOnDevice<D, E>, R: BuildOnDevice<D, E>> BuildOnDevice<D, E>
    for Split2<L, R>
{
    type Built = Split2<L::Built, R::Built>;
}

impl<I: WithEmptyTape, L: Module<I>, R: Module<I, Error = L::Error>> Module<I> for Split2<L, R> {
    type Output = (L::Output, R::Output);
    type Error = L::Error;

    fn try_forward(&self, input: I) -> Result<Self::Output, Self::Error> {
        let r_out = self.r.try_forward(input.with_empty_tape())?;
        let l_out = self.l.try_forward(input)?;
        Ok((l_out, r_out))
    }
}

impl<I: WithEmptyTape, L: ModuleMut<I>, R: ModuleMut<I, Error = L::Error>> ModuleMut<I>
    for Split2<L, R>
{
    type Output = (L::Output, R::Output);
    type Error = L::Error;

    fn try_forward_mut(&mut self, input: I) -> Result<Self::Output, Self::Error> {
        let r_out = self.r.try_forward_mut(input.with_empty_tape())?;
        let l_out = self.l.try_forward_mut(input)?;
        Ok((l_out, r_out))
    }
}

#[derive(Clone)]
pub struct Upscale2DResidual<M>(M);

impl<E: Dtype, D: Device<E>, M: TensorCollection<E, D>> TensorCollection<E, D>
    for Upscale2DResidual<M>
{
    type To<E2: Dtype, D2: Device<E2>> = Upscale2DResidual<M::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(Self::module("0", |s| &s.0, |s| &mut s.0), Upscale2DResidual)
    }
}

impl<E: Dtype, D: Device<E>, M: BuildOnDevice<D, E>> BuildOnDevice<D, E> for Upscale2DResidual<M> {
    type Built = Upscale2DResidual<M::Built>;
}

impl<I: WithEmptyTape + TryAdd + HasShape, M: Module<I, Error = I::Err>> Module<I>
    for Upscale2DResidual<M>
where
    M::Output: GenericUpscale2D<Bilinear> + TryUpscale2D + HasErr<Err = I::Err>,
    <M::Output as GenericUpscale2D<Bilinear>>::Output<usize, usize>:
        HasShape<WithShape<I::Shape> = I> + RealizeTo + std::fmt::Debug,
    <<M::Output as GenericUpscale2D<Bilinear>>::Output<usize, usize> as HasShape>::Shape:
        Shape<Concrete = <I::Shape as Shape>::Concrete>,
{
    type Output = I;
    type Error = M::Error;

    fn try_forward(&self, input: I) -> Result<I, Self::Error> {
        let residual = input.with_empty_tape();
        let shape = input.shape().concrete();
        let height = shape[I::Shape::NUM_DIMS - 2];
        let width = shape[I::Shape::NUM_DIMS - 1];
        let output = self
            .0
            .try_forward(input)?
            .try_upscale2d_like(Bilinear, height, width)?
            .realize::<I::Shape>()
            .unwrap();
        output.try_add(residual)
    }
}

impl<I: WithEmptyTape + TryAdd + HasShape, M: ModuleMut<I, Error = I::Err>> ModuleMut<I>
    for Upscale2DResidual<M>
where
    M::Output: GenericUpscale2D<Bilinear> + TryUpscale2D + HasErr<Err = I::Err>,
    <M::Output as GenericUpscale2D<Bilinear>>::Output<usize, usize>:
        HasShape<WithShape<I::Shape> = I> + RealizeTo + std::fmt::Debug,
    <<M::Output as GenericUpscale2D<Bilinear>>::Output<usize, usize> as HasShape>::Shape:
        Shape<Concrete = <I::Shape as Shape>::Concrete>,
{
    type Output = I;
    type Error = M::Error;

    fn try_forward_mut(&mut self, input: I) -> Result<I, Self::Error> {
        let residual = input.with_empty_tape();
        let shape = input.shape().concrete();
        let height = shape[I::Shape::NUM_DIMS - 2];
        let width = shape[I::Shape::NUM_DIMS - 1];
        let output = self
            .0
            .try_forward_mut(input)?
            .try_upscale2d_like(Bilinear, height, width)?
            .realize::<I::Shape>()
            .unwrap();
        output.try_add(residual)
    }
}
