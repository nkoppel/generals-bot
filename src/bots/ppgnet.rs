use dfdx::prelude::*;

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

type Head = (Conv2D<CHANNELS, 128, 3, 1, 1>, BatchNorm2D<128>, ReLU);

type Block<const C: usize, const D: usize, const P: usize> = (
    Residual<(
        Conv2D<C, C, D, 1, P>,
        BatchNorm2D<C>,
        ReLU,
        Conv2D<C, C, D, 1, P>,
        BatchNorm2D<C>,
    )>,
    ReLU,
);

type Downsample<const I: usize, const O: usize> = (Conv2D<I, O, 3, 1, 1>, BatchNorm2D<O>, ReLU);

pub type BigNet = (
    Head,
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
