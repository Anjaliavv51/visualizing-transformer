const GROUPS = [
  { name: '01. Foundations', ids: [1, 2], c: '--fill-softmax' },
  { name: '02. Attention', ids: [3, 4], c: '--fill-attn' },
  { name: '03. Architecture', ids: [5, 6, 7], c: '--fill-ffn' },
  { name: '04. Hands-on', ids: [8], c: '--fill-linear' }
];
const visited = new Set();
const SHORT = ['Bottlenecks','Paradigm','Scaled dot','Multi-head','Embeddings','Encoder','Decoder','Hands-on'];
const $id = (i) => document.getElementById(i);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
/**
 * ============================================================================
 * TRANSFORMER ARCHITECTURE EXPLAINER - APPLICATION ENGINE
 * Faithful to "Attention Is All You Need" (Vaswani et al., 2017)
 * ============================================================================
 */

// Model Configuration Presets
const PRESETS = {
  base: {
    name: "Transformer Base (Paper)",
    N: 6,
    d_model: 512,
    d_ff: 2048,
    h: 8,
    d_k: 64,
    d_v: 64,
    n: 10,
    m: 8,
    V: 37000,
    batch: 1,
    p_drop: 0.1
  },
  big: {
    name: "Transformer Big (Paper)",
    N: 6,
    d_model: 1024,
    d_ff: 4096,
    h: 16,
    d_k: 64,
    d_v: 64,
    n: 10,
    m: 8,
    V: 37000,
    batch: 1,
    p_drop: 0.3
  },
  gpt2_small: {
    name: "GPT-2 Small (Decoder-only)",
    N: 12,
    d_model: 768,
    d_ff: 3072,
    h: 12,
    d_k: 64,
    d_v: 64,
    n: 16,
    m: 16,
    V: 50257,
    batch: 1,
    p_drop: 0.1
  },
  tiny: {
    name: "Educational Tiny Model",
    N: 2,
    d_model: 64,
    d_ff: 256,
    h: 4,
    d_k: 16,
    d_v: 16,
    n: 6,
    m: 5,
    V: 1000,
    batch: 1,
    p_drop: 0.1
  }
};

// Current Application State
const state = {
  currentChapter: 1,
  totalChapters: 8,
  activeView: 'blueprint', // 'blueprint' | 'attention' | 'dimensions' | 'benchmark'
  presetKey: 'base',
  config: { ...PRESETS.base },
  activeComponentKey: 'enc_self_attn',
  isScaled: true,
  isMasked: false,
  selectedHead: 0,
  attentionSentence: "The law will never be perfect , but its application should be just"
};

// Component Metadata for Deep Inspection Drawer
const ARCH_COMPONENTS = {
  input_embed: {
    name: "Input Embedding & Scale",
    type: "Embedding",
    section: "Section 3.4",
    equation: "x_{embed} = \\text{Embedding}(x) \\times \\sqrt{d_{model}}",
    description: "Converts discrete input token IDs (from vocabulary V) into continuous vectors of dimension d_model. Crucially, Vaswani et al. multiply these weights by sqrt(d_model) so that the variance of the learned token embedding is on the same scale as the unit-variance sinusoidal positional encodings.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => cfg.V * cfg.d_model,
    paramFormula: "V \\times d_{model}"
  },
  input_pe: {
    name: "Positional Encoding (Sinusoidal)",
    type: "Encoding",
    section: "Section 3.5",
    equation: "PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d_{model}}), \\quad PE_{(pos, 2i+1)} = \\cos(pos/10000^{2i/d_{model}})",
    description: "Since the Transformer contains neither recurrence nor convolution, positional encodings are injected so the model can distinguish word order. Sinusoidal functions of differing wavelengths (from 2π to 10000·2π) allow the model to easily learn relative position offsets: for any fixed offset k, PE_{pos+k} is a linear function of PE_{pos}.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 0,
    paramFormula: "0 \\text{ (Fixed parameter-free)}"
  },
  enc_self_attn: {
    name: "Encoder Multi-Head Self-Attention",
    type: "Attention",
    section: "Section 3.2.2 & 3.2.3",
    equation: "\\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1, \\dots, head_h)W^O, \\quad head_i = \\text{softmax}\\left(\\frac{Q W_i^Q (K W_i^K)^T}{\\sqrt{d_k}}\\right) V W_i^V",
    description: "In the encoder, Queries, Keys, and Values all originate from the previous layer's representation. Every position in the sequence attends unrestrictedly to all other positions in parallel. Multi-head projection allows the model to jointly attend to information across diverse representation subspaces.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 4 * cfg.d_model * cfg.d_model,
    paramFormula: "4 \\times d_{model}^2 \\text{ (for } W^Q, W^K, W^V, W^O)"
  },
  enc_add_norm_1: {
    name: "Add & Layer Normalization (Sublayer 1)",
    type: "Norm",
    section: "Section 3.1",
    equation: "\\text{LayerNorm}(x + \\text{Sublayer}(x))",
    description: "A residual skip-connection is wrapped around the attention sublayer, followed by Layer Normalization across the d_model feature dimension. This facilitates stable gradient propagation through all N stacked layers.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 2 * cfg.d_model,
    paramFormula: "2 \\times d_{model} \\text{ (gain } \\gamma \\text{ and bias } \\beta)"
  },
  enc_ffn: {
    name: "Position-wise Feed-Forward Network",
    type: "FFN",
    section: "Section 3.3",
    equation: "\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2",
    description: "Applied to each position separately and identically. Consists of two linear transformations with a ReLU non-linear activation in between. The inner layer expands the representation to d_ff (typically 4x d_model) before projecting back.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model,
    paramFormula: "2 \\times d_{model} \\times d_{ff} + d_{ff} + d_{model}"
  },
  enc_add_norm_2: {
    name: "Add & Layer Normalization (Sublayer 2)",
    type: "Norm",
    section: "Section 3.1",
    equation: "\\text{LayerNorm}(x + \\text{FFN}(x))",
    description: "Residual skip-connection around the Feed-Forward sublayer followed by Layer Normalization.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Seq: ${cfg.n}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 2 * cfg.d_model,
    paramFormula: "2 \\times d_{model}"
  },
  dec_masked_self_attn: {
    name: "Masked Multi-Head Self-Attention",
    type: "Attention",
    section: "Section 3.2.3",
    equation: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V, \\quad M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}",
    description: "Modifies self-attention in the decoder by setting all illegal future connections (j > i) to negative infinity before softmax. This ensures predictions for position i depend strictly on known target tokens at positions less than i, preserving the autoregressive property.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 4 * cfg.d_model * cfg.d_model,
    paramFormula: "4 \\times d_{model}^2"
  },
  dec_cross_attn: {
    name: "Encoder-Decoder Cross-Attention",
    type: "Attention",
    section: "Section 3.2.3",
    equation: "Q = \\text{DecoderLayer}_{l-1}, \\quad K, V = \\text{EncoderOutput}",
    description: "The decoder queries the encoder's output representations. Queries come from the previous decoder sublayer, while Keys and Values come from the top of the encoder stack. This allows every target position in the decoder to attend across the entire source sequence.",
    calcInShape: (cfg) => `Q: [${cfg.batch}, ${cfg.m}, ${cfg.d_model}], \\; K,V: [${cfg.batch}, ${cfg.n}, ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 4 * cfg.d_model * cfg.d_model,
    paramFormula: "4 \\times d_{model}^2"
  },
  dec_ffn: {
    name: "Decoder Feed-Forward Network",
    type: "FFN",
    section: "Section 3.3",
    equation: "\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2",
    description: "Identical position-wise feed-forward architecture as in the encoder, applied to the decoder representation.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    params: (cfg) => 2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model,
    paramFormula: "2 \\times d_{model} \\times d_{ff} + d_{ff} + d_{model}"
  },
  linear_softmax: {
    name: "Linear Classifier & Softmax",
    type: "Output",
    section: "Section 3.4",
    equation: "P(y_t | y_{<t}, x) = \\text{softmax}(x_{dec} W_{proj} + b)",
    description: "Projects the decoder's final hidden state of dimension d_model up to the vocabulary dimension V. In Vaswani et al., the weight matrix W_proj is shared (tied) with the input and output embedding matrices, drastically reducing overall parameter count.",
    calcInShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, d_{model}: ${cfg.d_model}]`,
    calcOutShape: (cfg) => `[Batch: ${cfg.batch}, Target Seq: ${cfg.m}, Vocab: ${cfg.V}]`,
    params: (cfg) => cfg.V, // Tying weights means only bias is added
    paramFormula: "V \\text{ (biases, as weight matrix is tied with embeddings)}"
  }
};

// Guided Narrative Chapters Content
const CHAPTERS = [
  {
    id: 1,
    title: "1. The Pre-Attention Era & Recurrent Bottlenecks",
    subtitle: "Why sequence modeling needed a radical architectural breakthrough",
    tag: "Historical Motivation & Bottlenecks",
    content: `
      <div class="story-section">
        <h4>The Sequential Computation Ceiling</h4>
        <p>Before 2017, the undisputed kings of Natural Language Processing and Machine Translation were <strong>Recurrent Neural Networks (RNNs)</strong>, specifically <strong>LSTMs</strong> and <strong>GRUs</strong>.</p>
        <p>Recurrent models compute hidden states sequentially along the positions of the input sequence:</p>
        <div class="math-block">h_t = f(h_{t-1}, x_t)</div>
        <p>Because each step $t$ strictly requires the hidden state $h_{t-1}$ from the previous step, <strong>parallelization within a training sequence is mathematically impossible</strong>. As sequence length $n$ grows, this creates a catastrophic training bottleneck on GPUs.</p>
      </div>

      <div class="callout amber">
        <div class="callout-title">⚠️ The Information Bottleneck & Vanishing Gradients</div>
        <p>In traditional sequence-to-sequence RNNs, the entire input sentence was compressed into a single fixed-length vector at step $n$. Long-range dependencies decayed because signals had to traverse $O(n)$ sequential transitions, causing vanishing gradients.</p>
      </div>

      <div class="story-section">
        <h4>Table 1 from the Paper: Architectural Comparison</h4>
        <p>Vaswani et al. formulated the fundamental efficiency criteria comparing Self-Attention against Recurrent and Convolutional models:</p>
        <ul>
          <li><strong>Sequential Operations:</strong> Recurrent is $O(n)$ (sequential), while Self-Attention is $O(1)$ (fully parallel).</li>
          <li><strong>Maximum Path Length:</strong> Distance between any two arbitrary tokens in Self-Attention is $O(1)$ versus $O(n)$ in RNNs.</li>
          <li><strong>Per-Layer Complexity:</strong> Self-Attention costs $O(n^2 \\cdot d)$. Since in language sequences typically $n < d$, Self-Attention is computationally faster!</li>
        </ul>
      </div>
    `,
    suggestedView: 'blueprint'
  },
  {
    id: 2,
    title: "2. The Paradigm Shift: 'Attention Is All You Need'",
    subtitle: "Dispensing with recurrence and convolutions entirely",
    tag: "Core Concept",
    content: `
      <div class="story-section">
        <h4>Eliminating the Recurrence Loop</h4>
        <p>The core thesis of the landmark 2017 paper is encapsulated in its title: <em>Attention Is All You Need</em>. Instead of using attention merely as an auxiliary connection between an RNN encoder and an RNN decoder, the authors eliminated recurrence completely.</p>
        <p>Every token in the input sequence can now interact directly with every other token in a single computational step, reducing the maximum path length between distant words to <strong>O(1)</strong>.</p>
      </div>

      <div class="callout cyan">
        <div class="callout-title">💡 The Q, K, V Mental Model (Database Retrieval Analogy)</div>
        <p>Vaswani et al. formalized attention as a fuzzy, differentiable database lookup:</p>
        <ul>
          <li><strong>Query (Q):</strong> What a token is actively searching for (e.g., a verb looking for its subject).</li>
          <li><strong>Key (K):</strong> The label or attribute describing what each token has to offer.</li>
          <li><strong>Value (V):</strong> The actual information content that gets retrieved.</li>
        </ul>
      </div>

      <div class="story-section">
        <h4>Attention as a Weighted Sum</h4>
        <p>The attention output is a weighted sum of the Values, where the weight assigned to each Value is computed by a compatibility function (dot product) of the Query with the corresponding Key.</p>
        <div class="math-block">\\text{Output} = \\sum_{j} \\alpha_{i,j} V_j</div>
      </div>
    `,
    suggestedView: 'blueprint'
  },
  {
    id: 3,
    title: "3. Scaled Dot-Product Attention in Mathematical Detail",
    subtitle: "Why the square-root scaling factor √dk is non-negotiable",
    tag: "Equation 1 & Gradient Stability",
    content: `
      <div class="story-section">
        <h4>Equation 1: Scaled Dot-Product Attention</h4>
        <p>In practice, queries, keys, and values are packed simultaneously into matrices $Q$, $K$, and $V$:</p>
        <div class="math-block">\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V</div>
      </div>

      <div class="callout purple">
        <div class="callout-title">🔬 Why Scale by 1 / √dk? (Footnote 4 of Paper)</div>
        <p>Assume the components of query $q$ and key $k$ are independent random variables with mean 0 and variance 1.</p>
        <p>Their dot product $q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i$ has mean 0 and <strong>variance $d_k$</strong>.</p>
        <p>For large key dimensions like $d_k = 64$, the standard deviation is $\\sqrt{64} = 8$. As dot product magnitudes explode into large positive and negative numbers, the <strong>Softmax function saturates</strong> into extreme regions with near-zero gradients:</p>
        <div class="math-block">\\sigma(z)_i = \\frac{e^{z_i}}{\\sum e^{z_j}} \\implies \\frac{\\partial \\sigma_i}{\\partial z_j} \\approx 0</div>
        <p>Dividing by $\\sqrt{d_k}$ normalizes the variance back to 1, guaranteeing robust gradient flow during backpropagation.</p>
      </div>

      <div class="story-section">
        <h4>Dot-Product vs Additive Attention</h4>
        <p>While additive attention (Bahdanau et al.) and dot-product attention share similar theoretical complexity, dot-product attention is vastly faster and more memory-efficient because it leverages highly optimized matrix multiplication routines (BLAS / cuBLAS).</p>
      </div>
    `,
    suggestedView: 'attention'
  },
  {
    id: 4,
    title: "4. Multi-Head Attention: Subspaces & Parallel Heads",
    subtitle: "Why 8 specialized attention heads vastly outperform a single head",
    tag: "Section 3.2.2 & Figure 2",
    content: `
      <div class="story-section">
        <h4>The Limitation of a Single Attention Head</h4>
        <p>If we compute attention with a single head of size $d_{model}$, averaging over all positions inhibits the model from simultaneously attending to information from different representation subspaces at different positions.</p>
      </div>

      <div class="callout emerald">
        <div class="callout-title">🧠 Multi-Head Formulation</div>
        <div class="math-block">\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O</div>
        <div class="math-block">\\text{head}_i = \\text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)</div>
        <p>Where learned projection matrices are:</p>
        <ul>
          <li>$W_i^Q \\in \\mathbb{R}^{d_{model} \\times d_k}$</li>
          <li>$W_i^K \\in \\mathbb{R}^{d_{model} \\times d_k}$</li>
          <li>$W_i^V \\in \\mathbb{R}^{d_{model} \\times d_v}$</li>
          <li>$W^O \\in \\mathbb{R}^{h d_v \\times d_{model}}$</li>
        </ul>
      </div>

      <div class="story-section">
        <h4>The Dimension Split (dk = dv = dmodel / h = 64)</h4>
        <p>In the paper's base model, $h = 8$ parallel heads are used. For each head:</p>
        <div class="math-block">d_k = d_v = \\frac{d_{model}}{h} = \\frac{512}{8} = 64</div>
        <p>Because of this dimensional reduction, the total computational cost of 8 heads is <strong>virtually identical</strong> to that of single-head attention with full dimensionality, while granting rich representational diversity!</p>
      </div>
    `,
    suggestedView: 'attention'
  },
  {
    id: 5,
    title: "5. Input Embeddings & Sinusoidal Positional Encoding",
    subtitle: "Injecting geometric order without recurrence",
    tag: "Section 3.4 & 3.5",
    content: `
      <div class="story-section">
        <h4>Order Invariance of Self-Attention</h4>
        <p>A self-attention layer is permutation-equivariant: without positional cues, changing word order in a sentence produces identical representations. In natural language, however, <em>"The dog bit the man"</em> differs vastly from <em>"The man bit the dog"</em>.</p>
      </div>

      <div class="callout cyan">
        <div class="callout-title">📐 The Sinusoidal Formulation</div>
        <div class="math-block">PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)</div>
        <div class="math-block">PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)</div>
        <p>Each dimension $i$ corresponds to a sinusoid whose wavelengths form a geometric progression from $2\\pi$ to $10000 \\cdot 2\\pi$.</p>
      </div>

      <div class="story-section">
        <h4>Why Sinusoids?</h4>
        <p>The authors hypothesized sinusoids allow the model to learn relative position offsets easily, because for any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$ using standard trigonometric angle addition formulas.</p>
        <p>Moreover, unlike learned positional embeddings, sinusoids allow the model to extrapolate to sequence lengths longer than those observed during training!</p>
      </div>

      <div class="callout amber">
        <div class="callout-title">⚡ The √dmodel Embedding Multiplier</div>
        <p>In Section 3.4, the authors note that token embeddings are multiplied by $\\sqrt{d_{model}}$ (e.g. $\\sqrt{512} \\approx 22.6$) before adding positional encodings. This prevents the unit-variance positional encodings from overpowering the learned word semantic vectors!</p>
      </div>
    `,
    suggestedView: 'dimensions'
  },
  {
    id: 6,
    title: "6. Inside the Encoder Stack: N = 6 Layers",
    subtitle: "Layer-by-layer anatomy of continuous representation building",
    tag: "Section 3.1 & Figure 1 (Left)",
    content: `
      <div class="story-section">
        <h4>Encoder Architecture Overview</h4>
        <p>The encoder is composed of a stack of $N = 6$ identical layers. Each layer contains two distinct sub-layers:</p>
        <ol>
          <li><strong>Multi-Head Self-Attention mechanism</strong></li>
          <li><strong>Position-wise Fully Connected Feed-Forward Network</strong></li>
        </ol>
      </div>

      <div class="callout emerald">
        <div class="callout-title">🔗 Residuals & LayerNorm: LayerNorm(x + Sublayer(x))</div>
        <p>Around each sub-layer, a residual skip connection is applied, followed by Layer Normalization:</p>
        <div class="math-block">\\text{Output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))</div>
        <p>To facilitate these additions, all sub-layers as well as embedding layers produce outputs of constant dimension $d_{model} = 512$.</p>
      </div>

      <div class="story-section">
        <h4>Position-wise Feed-Forward Network (FFN)</h4>
        <div class="math-block">\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2</div>
        <p>The inner dimension expands to $d_{ff} = 2048$ ($4 \\times d_{model}$). This can also be interpreted as two convolutions with kernel size 1.</p>
        <p>The final output of the 6th encoder layer is a set of contextualized continuous vectors $z = (z_1, \\dots, z_n)$ representing the full source sentence.</p>
      </div>
    `,
    suggestedView: 'blueprint'
  },
  {
    id: 7,
    title: "7. Inside the Decoder Stack: Autoregression & Causal Masking",
    subtitle: "Preventing leftward information flow and cross-attending to encoder",
    tag: "Section 3.1 & Figure 1 (Right)",
    content: `
      <div class="story-section">
        <h4>The 3 Sub-layers of the Decoder</h4>
        <p>The decoder is also composed of a stack of $N = 6$ identical layers, but inserts a third sublayer:</p>
        <ol>
          <li><strong>Masked Multi-Head Self-Attention:</strong> Attends only to past and current output positions.</li>
          <li><strong>Multi-Head Cross-Attention:</strong> Attends over the output of the encoder stack.</li>
          <li><strong>Position-wise Feed-Forward Network</strong></li>
        </ol>
      </div>

      <div class="callout purple">
        <div class="callout-title">🔒 The Causal Attention Mask (Preserving Autoregression)</div>
        <p>During training, the decoder receives the target sequence shifted right by one position. To prevent position $i$ from 'peeking' into future tokens $j > i$, a lower-triangular causal mask is applied before softmax:</p>
        <div class="math-block">M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}</div>
        <p>Since $e^{-\\infty} = 0$, attention weights to future positions are strictly zero!</p>
      </div>

      <div class="story-section">
        <h4>Encoder-Decoder Cross-Attention</h4>
        <p>In this sublayer, the <strong>Queries ($Q$)</strong> come from the previous decoder sublayer, while the <strong>Keys ($K$) and Values ($V$)</strong> come from the output of the encoder stack.</p>
        <p>This allows every generated target token to query and attend across the entire source sequence representation!</p>
      </div>
    `,
    suggestedView: 'blueprint'
  },
  {
    id: 8,
    title: "8. Live Architecture Blueprint & Dimension Laboratory",
    subtitle: "Custom input shapes, tensor propagation, and parameter budgeting",
    tag: "Interactive Studio",
    content: `
      <div class="story-section">
        <h4>Full Model Dimension Propagation</h4>
        <p>In modern language engineering, understanding how tensor shapes transform across each layer is critical for memory planning and FLOP estimation.</p>
        <p>Use the controls below or in the right pane to adjust sequence lengths ($n, m$), embedding dimension ($d_{model}$), heads ($h$), and vocabulary size ($V$).</p>
      </div>

      <div class="dimension-quick-controls">
        <div class="dimension-control-row">
          <div class="dimension-label">
            Source Sequence (n)
            <small>Input sentence token count</small>
          </div>
          <div class="dimension-input-group">
            <input type="range" id="quick_n" min="2" max="64" value="10">
            <span class="dimension-val-badge" id="badge_quick_n">10</span>
          </div>
        </div>

        <div class="dimension-control-row">
          <div class="dimension-label">
            Target Sequence (m)
            <small>Output sentence token count</small>
          </div>
          <div class="dimension-input-group">
            <input type="range" id="quick_m" min="2" max="64" value="8">
            <span class="dimension-val-badge" id="badge_quick_m">8</span>
          </div>
        </div>

        <div class="dimension-control-row">
          <div class="dimension-label">
            Model Dim (d_model)
            <small>Internal feature dimension</small>
          </div>
          <div class="dimension-input-group">
            <input type="range" id="quick_dmodel" min="64" max="2048" step="64" value="512">
            <span class="dimension-val-badge" id="badge_quick_dmodel">512</span>
          </div>
        </div>

        <div class="dimension-control-row">
          <div class="dimension-label">
            Attention Heads (h)
            <small>Number of parallel attention heads</small>
          </div>
          <div class="dimension-input-group">
            <input type="range" id="quick_h" min="1" max="32" step="1" value="8">
            <span class="dimension-val-badge" id="badge_quick_h">8</span>
          </div>
        </div>
      </div>

      <div class="callout emerald">
        <div class="callout-title">📊 Click Any Diagram Component</div>
        <p>Switch to the <strong>Architecture Blueprint</strong> tab on the right and click any block (Multi-Head Attention, Add & Norm, Feed-Forward, Positional Encoding) to open the Deep Inspector Drawer with exact live tensor shapes and parameter formulas!</p>
      </div>
    `,
    suggestedView: 'dimensions'
  }
];

// Calculation Functions
function computeDerivedParams(cfg) {
  const d_k = Math.floor(cfg.d_model / cfg.h);
  const d_v = d_k;

  // Single Encoder Layer Params
  // 1. Self-Attention: W_Q, W_K, W_V each (d_model x (h*d_k) = d_model^2) + bias, W_O (d_model^2) + bias
  const attn_weights = 4 * cfg.d_model * cfg.d_model;
  const attn_biases = 4 * cfg.d_model;
  const enc_self_attn_params = attn_weights + attn_biases;

  // 2. Add & Norm (2 norm layers per encoder layer: gamma & beta each of length d_model)
  const norm_params = 2 * cfg.d_model;

  // 3. FFN: W1 (d_model x d_ff) + b1 (d_ff) + W2 (d_ff x d_model) + b2 (d_model)
  const ffn_params = 2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model;

  const single_encoder_params = enc_self_attn_params + (norm_params * 2) + ffn_params;
  const total_encoder_params = single_encoder_params * cfg.N;

  // Single Decoder Layer Params
  // 1. Masked Self-Attention
  const dec_masked_attn_params = enc_self_attn_params;
  // 2. Cross-Attention
  const dec_cross_attn_params = enc_self_attn_params;
  // 3. FFN
  const dec_ffn_params = ffn_params;
  // 3 Norm layers in decoder
  const dec_norms_params = norm_params * 3;

  const single_decoder_params = dec_masked_attn_params + dec_cross_attn_params + dec_ffn_params + dec_norms_params;
  const total_decoder_params = single_decoder_params * cfg.N;

  // Embeddings (weight tying between input embed, output embed, and pre-softmax linear)
  // Vaswani et al. share the same weight matrix between input embedding, output embedding, and pre-softmax linear!
  const shared_embedding_params = cfg.V * cfg.d_model;
  const pre_softmax_bias = cfg.V;

  const grand_total_params = shared_embedding_params + pre_softmax_bias + total_encoder_params + total_decoder_params;

  return {
    d_k,
    d_v,
    enc_self_attn_params,
    single_encoder_params,
    total_encoder_params,
    single_decoder_params,
    total_decoder_params,
    shared_embedding_params,
    grand_total_params
  };
}

// DOM Elements
const elements = {
  chapterPills: document.getElementById('chapter_pills'),
  storyContent: document.getElementById('story_content'),
  storyTitle: document.getElementById('story_title'),
  storySummary: document.getElementById('story_summary'),
  stepIndicator: document.getElementById('step_indicator'),
  stepTag: document.getElementById('step_tag'),
  btnPrev: document.getElementById('btn_prev'),
  btnNext: document.getElementById('btn_next'),
  stageContent: document.getElementById('stage_content'),
  presetDropdown: document.getElementById('preset_dropdown'),
  tabBlueprint: document.getElementById('tab_blueprint'),
  tabAttention: document.getElementById('tab_attention'),
  tabDimensions: document.getElementById('tab_dimensions'),
  tabBenchmark: document.getElementById('tab_benchmark'),
  inspectorDrawer: document.getElementById('inspector_drawer'),
  btnCloseDrawer: document.getElementById('btn_close_drawer'),
  metricTotalParams: document.getElementById('metric_total_params'),
  metricDk: document.getElementById('metric_dk'),
  metricComplexity: document.getElementById('metric_complexity')
};

// Initialize Application
function initApp() {
  renderChapterPills();
  setupShellUI();
  loadChapter(state.currentChapter);
  setupEventListeners();
  updateLiveMetrics();
}

// Render chapter stepper + course tree (textContent only, no HTML injection)
function renderChapterPills() {
  elements.chapterPills.replaceChildren();
  CHAPTERS.forEach((ch) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chapter-pill';
    const n = document.createElement('span'); n.className = 'num'; n.textContent = ch.id;
    const l = document.createElement('span'); l.className = 'lbl'; l.textContent = SHORT[ch.id - 1] || ch.tag;
    btn.append(n, l);
    btn.title = ch.title;
    btn.addEventListener('click', () => loadChapter(ch.id));
    elements.chapterPills.appendChild(btn);
  });
  renderTree('');
}

function renderTree(query) {
  const tree = $id('course_tree');
  tree.replaceChildren();
  const needle = String(query || '').trim().toLowerCase();
  GROUPS.forEach((g) => {
    const items = g.ids.map((id) => CHAPTERS.find((c) => c.id === id))
      .filter((c) => !needle || (c.title + ' ' + c.tag + ' ' + c.subtitle).toLowerCase().includes(needle));
    if (!items.length) return;
    const wrap = document.createElement('div'); wrap.className = 'tree-group';
    const head = document.createElement('div'); head.className = 'tree-head';
    const dot = document.createElement('span'); dot.className = 'folder'; dot.style.background = `var(${g.c})`;
    const nm = document.createElement('span'); nm.textContent = g.name;
    const ct = document.createElement('span'); ct.className = 'count';
    ct.textContent = `${items.filter((c) => visited.has(c.id)).length}/${items.length}`;
    head.append(dot, nm, ct);
    const list = document.createElement('div'); list.className = 'tree-items';
    items.forEach((c) => {
      const b = document.createElement('button'); b.type = 'button';
      b.className = 'tree-item' + (c.id === state.currentChapter ? ' active' : '');
      b.textContent = c.title.replace(/^\d+\.\s*/, '');
      b.addEventListener('click', () => { loadChapter(c.id); closeMenu(); });
      list.appendChild(b);
    });
    wrap.append(head, list); tree.appendChild(wrap);
  });
  if (!tree.children.length) { const p = document.createElement('p'); p.className = 'count'; p.textContent = 'No chapters match. Try another word.'; tree.appendChild(p); }
}

function closeMenu() {
  $id('sidebar').classList.remove('open'); $id('backdrop').hidden = true; $id('btn_menu').setAttribute('aria-expanded', 'false');
}

function setupShellUI() {
  $id('btn_menu').addEventListener('click', () => {
    const open = $id('sidebar').classList.toggle('open');
    $id('backdrop').hidden = !open; $id('btn_menu').setAttribute('aria-expanded', String(open));
  });
  $id('backdrop').addEventListener('click', closeMenu);
  const search = $id('course_search');
  search.addEventListener('input', () => renderTree(search.value));
  search.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Escape') { search.value = ''; renderTree(''); search.blur(); } });
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); search.focus(); }
    if (e.key === 'Escape') closeMenu();
  });
}

// Load Chapter
function loadChapter(chapterId) {
  state.currentChapter = chapterId;
  const chapter = CHAPTERS.find(c => c.id === chapterId) || CHAPTERS[0];

  elements.stepIndicator.textContent = `Lesson ${chapter.id} of ${state.totalChapters}`;
  elements.stepTag.textContent = chapter.tag;
  elements.storyTitle.textContent = chapter.title;
  elements.storySummary.textContent = chapter.subtitle;
  elements.storyContent.innerHTML = chapter.content;
  visited.add(chapter.id);
  const grp = GROUPS.find((x) => x.ids.includes(chapter.id));
  $id('crumb_group').textContent = grp ? grp.name : '';
  $id('step_count').textContent = `CHAPTER ${chapter.id} / ${state.totalChapters}`;
  const nWords = (elements.storyContent.textContent || '').trim().split(/\s+/).length;
  const heads = [...elements.storyContent.querySelectorAll('h4')];
  $id('pill_time').textContent = `${Math.max(1, Math.round(nWords / 180))} min read`;
  $id('pill_sections').textContent = `${heads.length} sections`;
  const ll = $id('learn_list'); ll.replaceChildren();
  heads.forEach((h4) => {
    const row = document.createElement('label'); row.className = 'learn-item';
    const cb = document.createElement('input'); cb.type = 'checkbox';
    const sp = document.createElement('span'); sp.textContent = h4.textContent;
    row.append(cb, sp); ll.appendChild(row);
  });
  $id('progress_val').textContent = `${visited.size} / ${state.totalChapters}`;
  renderTree($id('course_search').value);
  const sb = document.querySelector('.step-body'); if (sb) sb.scrollTop = 0;

  elements.btnPrev.disabled = chapterId <= 1;
  elements.btnNext.disabled = chapterId >= state.totalChapters;

  // Update pill states
  const pills = elements.chapterPills.querySelectorAll('.chapter-pill');
  pills.forEach((p, idx) => {
    p.classList.toggle('active', idx + 1 === chapterId);
  });

  // Switch view if suggested
  if (chapter.suggestedView) {
    switchView(chapter.suggestedView, false);
  }

  // Bind quick controls if on chapter 8
  if (chapterId === 8) {
    setupChapter8Controls();
  }

  renderStageView();
  renderMath();
}

function renderInlineMath() {
  if (!window.katex) return;
  document.querySelectorAll('#story_content, #stage_content').forEach((root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const n = walker.currentNode;
      if (/\$[^$\n]+\$/.test(n.nodeValue) && !n.parentElement.closest('.math-block, script, style, textarea, input, option')) nodes.push(n);
    }
    nodes.forEach((n) => {
      const frag = document.createDocumentFragment();
      n.nodeValue.split(/(\$[^$\n]+\$)/).forEach((part) => {
        if (/^\$[^$\n]+\$$/.test(part)) {
          const span = document.createElement('span');
          try { katex.render(part.slice(1, -1), span, { throwOnError: false }); frag.appendChild(span); }
          catch (e) { frag.appendChild(document.createTextNode(part)); }
        } else if (part) frag.appendChild(document.createTextNode(part));
      });
      n.parentNode.replaceChild(frag, n);
    });
  });
}

function renderMath() {
  renderInlineMath();
  if (window.katex) {
    document.querySelectorAll('.math-block').forEach(el => {
      if (!el.dataset.rendered) {
        const raw = el.textContent.trim();
        try {
          katex.render(raw, el, { throwOnError: false, displayMode: true });
          el.dataset.rendered = "true";
        } catch (e) {
          // fallback to clean text
        }
      }
    });
  }
}

// Setup Chapter 8 Quick Sliders
function setupChapter8Controls() {
  const quick_n = document.getElementById('quick_n');
  const quick_m = document.getElementById('quick_m');
  const quick_dmodel = document.getElementById('quick_dmodel');
  const quick_h = document.getElementById('quick_h');

  if (quick_n) {
    quick_n.value = state.config.n;
    document.getElementById('badge_quick_n').textContent = state.config.n;
    quick_n.addEventListener('input', (e) => {
      state.config.n = parseInt(e.target.value);
      document.getElementById('badge_quick_n').textContent = state.config.n;
      state.presetKey = 'custom';
      elements.presetDropdown.value = 'custom';
      updateLiveMetrics();
      renderStageView();
    });
  }

  if (quick_m) {
    quick_m.value = state.config.m;
    document.getElementById('badge_quick_m').textContent = state.config.m;
    quick_m.addEventListener('input', (e) => {
      state.config.m = parseInt(e.target.value);
      document.getElementById('badge_quick_m').textContent = state.config.m;
      state.presetKey = 'custom';
      elements.presetDropdown.value = 'custom';
      updateLiveMetrics();
      renderStageView();
    });
  }

  if (quick_dmodel) {
    quick_dmodel.value = state.config.d_model;
    document.getElementById('badge_quick_dmodel').textContent = state.config.d_model;
    quick_dmodel.addEventListener('input', (e) => {
      state.config.d_model = parseInt(e.target.value);
      state.config.d_ff = state.config.d_model * 4;
      document.getElementById('badge_quick_dmodel').textContent = state.config.d_model;
      state.presetKey = 'custom';
      elements.presetDropdown.value = 'custom';
      updateLiveMetrics();
      renderStageView();
    });
  }

  if (quick_h) {
    quick_h.value = state.config.h;
    document.getElementById('badge_quick_h').textContent = state.config.h;
    quick_h.addEventListener('input', (e) => {
      state.config.h = parseInt(e.target.value);
      document.getElementById('badge_quick_h').textContent = state.config.h;
      state.presetKey = 'custom';
      elements.presetDropdown.value = 'custom';
      updateLiveMetrics();
      renderStageView();
    });
  }
}

// Switch Right Stage Tab View
function switchView(viewName, triggerRender = true) {
  state.activeView = viewName;
  [elements.tabBlueprint, elements.tabAttention, elements.tabDimensions, elements.tabBenchmark].forEach(tab => {
    tab.classList.remove('active');
  });

  if (viewName === 'blueprint') elements.tabBlueprint.classList.add('active');
  if (viewName === 'attention') elements.tabAttention.classList.add('active');
  if (viewName === 'dimensions') elements.tabDimensions.classList.add('active');
  if (viewName === 'benchmark') elements.tabBenchmark.classList.add('active');

  if (triggerRender) {
    renderStageView();
  }
}

// Update Top Metrics Bar
function updateLiveMetrics() {
  const derived = computeDerivedParams(state.config);
  elements.metricTotalParams.textContent = formatParamCount(derived.grand_total_params);
  elements.metricDk.textContent = `${derived.d_k}`;
  elements.metricComplexity.textContent = `O(n²·d) = ${formatParamCount(state.config.n * state.config.n * state.config.d_model)}`;
}

function formatParamCount(num) {
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toString();
}

// Main Stage Renderer
function renderStageView() {
  updateLiveMetrics();

  if (state.activeView === 'blueprint') {
    renderBlueprintView();
  } else if (state.activeView === 'attention') {
    renderAttentionLabView();
  } else if (state.activeView === 'dimensions') {
    renderDimensionStudioView();
  } else if (state.activeView === 'benchmark') {
    renderBenchmarkView();
  }
}

// VIEW 1: Interactive Architecture Blueprint (Vaswani Figure 1)
function renderBlueprintView() {
  const cfg = state.config;
  const d = computeDerivedParams(cfg);

  elements.stageContent.innerHTML = `
    <div class="blueprint-wrapper">
      <div class="blueprint-toolbar">
        <div class="diagram-legend">
          <div class="legend-item"><span class="legend-dot" style="background:var(--encoder-color)"></span> Encoder Stack (N=${cfg.N})</div>
          <div class="legend-item"><span class="legend-dot" style="background:var(--decoder-color)"></span> Decoder Stack (N=${cfg.N})</div>
          <div class="legend-item"><span class="legend-dot" style="background:var(--attention-color)"></span> Multi-Head Attention</div>
          <div class="legend-item"><span class="legend-dot" style="background:var(--ffn-color)"></span> Feed Forward</div>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted)">
          💡 Click any block to inspect live tensor shapes & equations
        </div>
      </div>

      <div class="diagram-container">
        <!-- ENCODER COLUMN -->
        <div class="stack-column encoder">
          <div class="stack-title">
            <span>Encoder Stack</span>
            <span class="stack-badge">N = ${cfg.N} Layers</span>
          </div>

          <!-- Repeating N x Encoder -->
          <div class="repeating-stack">
            <span class="repeating-tag">N × ${cfg.N}</span>

            <!-- 1. Add & Norm 2 -->
            <div class="arch-block ${state.activeComponentKey === 'enc_add_norm_2' ? 'active' : ''}" data-key="enc_add_norm_2">
              <div class="arch-block-header">
                <span class="arch-block-title">Add & Norm</span>
                <span class="arch-block-type type-norm">LayerNorm</span>
              </div>
              <div class="arch-block-shape">
                <span>Out:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.n}, ${cfg.d_model}]</span>
              </div>
            </div>

            <div class="flow-arrow">▲</div>

            <!-- 2. Feed Forward -->
            <div class="arch-block ${state.activeComponentKey === 'enc_ffn' ? 'active' : ''}" data-key="enc_ffn">
              <div class="arch-block-header">
                <span class="arch-block-title">Feed Forward (FFN)</span>
                <span class="arch-block-type type-ffn">Linear + ReLU</span>
              </div>
              <div class="arch-block-shape">
                <span>Inner:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.n}, ${cfg.d_ff}]</span>
              </div>
            </div>

            <div class="flow-arrow">▲</div>

            <!-- 3. Add & Norm 1 -->
            <div class="arch-block ${state.activeComponentKey === 'enc_add_norm_1' ? 'active' : ''}" data-key="enc_add_norm_1">
              <div class="arch-block-header">
                <span class="arch-block-title">Add & Norm</span>
                <span class="arch-block-type type-norm">LayerNorm</span>
              </div>
              <div class="arch-block-shape">
                <span>Out:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.n}, ${cfg.d_model}]</span>
              </div>
            </div>

            <div class="flow-arrow">▲</div>

            <!-- 4. Multi-Head Self-Attention -->
            <div class="arch-block ${state.activeComponentKey === 'enc_self_attn' ? 'active' : ''}" data-key="enc_self_attn">
              <div class="arch-block-header">
                <span class="arch-block-title">Multi-Head Attention</span>
                <span class="arch-block-type type-attention">${cfg.h} Heads (dk=${d.d_k})</span>
              </div>
              <div class="arch-block-shape">
                <span>Q, K, V:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.n}, ${cfg.d_model}]</span>
              </div>
            </div>
          </div>

          <div class="flow-arrow">▲</div>

          <!-- Positional Encoding -->
          <div class="arch-block ${state.activeComponentKey === 'input_pe' ? 'active' : ''}" data-key="input_pe">
            <div class="arch-block-header">
              <span class="arch-block-title">Positional Encoding</span>
              <span class="arch-block-type type-embed">Sinusoidal</span>
            </div>
            <div class="arch-block-shape">
              <span>PE Shape:</span> <span class="shape-badge">[1, ${cfg.n}, ${cfg.d_model}]</span>
            </div>
          </div>

          <div class="flow-arrow">▲</div>

          <!-- Input Embedding -->
          <div class="arch-block ${state.activeComponentKey === 'input_embed' ? 'active' : ''}" data-key="input_embed">
            <div class="arch-block-header">
              <span class="arch-block-title">Input Embedding</span>
              <span class="arch-block-type type-embed">Tokens × √dmodel</span>
            </div>
            <div class="arch-block-shape">
              <span>Input:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.n}] → [${cfg.batch}, ${cfg.n}, ${cfg.d_model}]</span>
            </div>
          </div>
        </div>

        <!-- DECODER COLUMN -->
        <div class="stack-column decoder">
          <div class="stack-title">
            <span>Decoder Stack</span>
            <span class="stack-badge">N = ${cfg.N} Layers</span>
          </div>

          <!-- Linear & Softmax Output -->
          <div class="arch-block ${state.activeComponentKey === 'linear_softmax' ? 'active' : ''}" data-key="linear_softmax">
            <div class="arch-block-header">
              <span class="arch-block-title">Linear & Softmax</span>
              <span class="arch-block-type type-embed">Vocab Probabilities</span>
            </div>
            <div class="arch-block-shape">
              <span>Output:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.m}, Vocab: ${cfg.V}]</span>
            </div>
          </div>

          <div class="flow-arrow">▲</div>

          <!-- Repeating N x Decoder -->
          <div class="repeating-stack">
            <span class="repeating-tag">N × ${cfg.N}</span>

            <!-- Dec Feed Forward -->
            <div class="arch-block ${state.activeComponentKey === 'dec_ffn' ? 'active' : ''}" data-key="dec_ffn">
              <div class="arch-block-header">
                <span class="arch-block-title">Feed Forward (FFN)</span>
                <span class="arch-block-type type-ffn">Linear + ReLU</span>
              </div>
              <div class="arch-block-shape">
                <span>Inner:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.m}, ${cfg.d_ff}]</span>
              </div>
            </div>

            <div class="flow-arrow">▲</div>

            <!-- Encoder-Decoder Cross Attention -->
            <div class="arch-block ${state.activeComponentKey === 'dec_cross_attn' ? 'active' : ''}" data-key="dec_cross_attn">
              <div class="cross-attention-connector"></div>
              <div class="arch-block-header">
                <span class="arch-block-title">Cross-Attention</span>
                <span class="arch-block-type type-attention">Dec Q ← Enc K, V</span>
              </div>
              <div class="arch-block-shape">
                <span>Attn Map:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.h}, ${cfg.m}, ${cfg.n}]</span>
              </div>
            </div>

            <div class="flow-arrow">▲</div>

            <!-- Masked Multi-Head Self Attention -->
            <div class="arch-block ${state.activeComponentKey === 'dec_masked_self_attn' ? 'active' : ''}" data-key="dec_masked_self_attn">
              <div class="arch-block-header">
                <span class="arch-block-title">Masked Attention</span>
                <span class="arch-block-type type-attention">Causal Mask (-∞)</span>
              </div>
              <div class="arch-block-shape">
                <span>Attn Map:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.h}, ${cfg.m}, ${cfg.m}]</span>
              </div>
            </div>
          </div>

          <div class="flow-arrow">▲</div>

          <!-- Output Positional Encoding & Embedding -->
          <div class="arch-block" style="opacity: 0.85;">
            <div class="arch-block-header">
              <span class="arch-block-title">Output Embedding + PE</span>
              <span class="arch-block-type type-embed">Shifted Right</span>
            </div>
            <div class="arch-block-shape">
              <span>Target:</span> <span class="shape-badge">[${cfg.batch}, ${cfg.m}, ${cfg.d_model}]</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Attach click listener to all blocks
  const blocks = elements.stageContent.querySelectorAll('.arch-block');
  blocks.forEach(block => {
    block.addEventListener('click', () => {
      const key = block.getAttribute('data-key');
      if (key && ARCH_COMPONENTS[key]) {
        state.activeComponentKey = key;
        openInspector(key);
        // Highlight active
        blocks.forEach(b => b.classList.remove('active'));
        block.classList.add('active');
      }
    });
  });
}

// Open Inspector Drawer
function openInspector(key) {
  const comp = ARCH_COMPONENTS[key];
  if (!comp) return;

  const cfg = state.config;
  const inShape = comp.calcInShape(cfg);
  const outShape = comp.calcOutShape(cfg);
  const paramCount = comp.params(cfg);

  document.getElementById('drawer_title').textContent = comp.name;
  document.getElementById('drawer_type').textContent = `${comp.type} • ${comp.section}`;
  document.getElementById('drawer_desc').textContent = comp.description;
  document.getElementById('drawer_equation').textContent = comp.equation;
  document.getElementById('drawer_in_shape').textContent = inShape;
  document.getElementById('drawer_out_shape').textContent = outShape;
  document.getElementById('drawer_params').textContent = `${formatParamCount(paramCount)} (${paramCount.toLocaleString()} params)`;
  document.getElementById('drawer_param_formula').textContent = comp.paramFormula;

  elements.inspectorDrawer.classList.add('open');
  if (window.katex) {
    const eqEl = document.getElementById('drawer_equation');
    try {
      katex.render(comp.equation, eqEl, { throwOnError: false, displayMode: true });
    } catch(e) {}
  }
}

// VIEW 2: Scaled Dot-Product Attention & Heatmap Lab
function renderAttentionLabView() {
  const cfg = state.config;
  const dk = Math.floor(cfg.d_model / cfg.h);
  const words = state.attentionSentence.split(' ');
  const seqLen = words.length;

  elements.stageContent.innerHTML = `
    <div class="attention-lab-wrapper">
      <!-- Interactive Sentence input -->
      <div style="background:var(--bg-secondary); border:1px solid var(--border-subtle); border-radius:10px; padding:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
          <label style="font-weight:700; font-size:0.9rem; color:var(--text-main);">
            Interactive Sample Sentence (Vaswani Figure 4):
          </label>
          <div style="display:flex; gap:10px; align-items:center;">
            <label style="font-size:0.8rem; color:var(--text-muted); cursor:pointer;">
              <input type="checkbox" id="chk_scaling" ${state.isScaled ? 'checked' : ''}> Apply Scale 1/√dk (${(1/Math.sqrt(dk)).toFixed(3)})
            </label>
            <label style="font-size:0.8rem; color:var(--text-muted); cursor:pointer;">
              <input type="checkbox" id="chk_causal_mask" ${state.isMasked ? 'checked' : ''}> Causal Mask (Decoder mode)
            </label>
          </div>
        </div>
        <input type="text" id="input_sentence" value="${esc(state.attentionSentence)}" maxlength="120"
          style="width:100%; background:var(--bg-tertiary); border:1px solid var(--border-subtle); color:var(--text-main); padding:8px 12px; border-radius:6px; font-family:var(--font-sans); font-size:0.9rem;">
      </div>

      <!-- Q, K, V Projection Cards -->
      <div class="qkv-pipeline-grid">
        <div class="qkv-card q">
          <div class="qkv-title">
            <span>Query Matrix (Q)</span>
            <span class="shape-badge">[${seqLen}, ${dk}]</span>
          </div>
          <p style="font-size:0.75rem; color:var(--text-muted);">
            $Q = X \\cdot W_i^Q$. Represents what each token is looking for across other tokens.
          </p>
        </div>

        <div class="qkv-card k">
          <div class="qkv-title">
            <span>Key Matrix (K)</span>
            <span class="shape-badge">[${seqLen}, ${dk}]</span>
          </div>
          <p style="font-size:0.75rem; color:var(--text-muted);">
            $K = X \\cdot W_i^K$. Acts as the address key for matching with queries.
          </p>
        </div>

        <div class="qkv-card v">
          <div class="qkv-title">
            <span>Value Matrix (V)</span>
            <span class="shape-badge">[${seqLen}, ${dk}]</span>
          </div>
          <p style="font-size:0.75rem; color:var(--text-muted);">
            $V = X \\cdot W_i^V$. The contextual content vector retrieved by attention.
          </p>
        </div>
      </div>

      <!-- Step Calculation Pipeline -->
      <div class="matrix-calc-stage">
        <div class="step-card-box">
          <h5>Step 1: Compute Raw Dot Product (Q · K^T)</h5>
          <p style="font-size:0.85rem; color:var(--text-muted);">
            Matrix multiplication yields compatibility scores of shape <strong>[${seqLen}, ${seqLen}]</strong>.
            Without scaling, variance grows to $d_k = ${dk}$ ($std = ${(Math.sqrt(dk)).toFixed(1)}$).
          </p>
        </div>

        <div class="step-card-box" style="border-left-color:var(--accent-purple);">
          <h5>Step 2: Scaling by 1/√${dk} and Optional Masking</h5>
          <p style="font-size:0.85rem; color:var(--text-muted);">
            ${state.isScaled ? '✅ Scaled scores normalizes variance to 1.0, preventing softmax saturation.' : '❌ Scaling disabled: Extreme values saturate softmax into near-zero gradient regions!'}
            ${state.isMasked ? ' • Causal mask applied: Upper triangle set to -∞.' : ' • Bidirectional unmasked attention.'}
          </p>
        </div>

        <div class="step-card-box" style="border-left-color:var(--accent-emerald);">
          <h5>Step 3: Attention Weight Heatmap: softmax((QK^T)/√dk)</h5>
          <div id="heatmap_render_target" class="heatmap-container">
            <!-- Heatmap will be generated here -->
          </div>
        </div>
      </div>
    </div>
  `;

  // Attach controls
  const chkScale = document.getElementById('chk_scaling');
  const chkMask = document.getElementById('chk_causal_mask');
  const inputSent = document.getElementById('input_sentence');

  chkScale.addEventListener('change', (e) => {
    state.isScaled = e.target.checked;
    renderAttentionLabView();
  });

  chkMask.addEventListener('change', (e) => {
    state.isMasked = e.target.checked;
    renderAttentionLabView();
  });

  inputSent.addEventListener('change', (e) => {
    state.attentionSentence = e.target.value.trim() || "The animal didn't cross the street because it was too tired";
    renderAttentionLabView();
  });

  generateSimulatedAttentionHeatmap(words, dk, state.isScaled, state.isMasked);
}

// Generate Dynamic Heatmap Matrix
function generateSimulatedAttentionHeatmap(words, dk, isScaled, isMasked) {
  const container = document.getElementById('heatmap_render_target');
  if (!container) return;

  const n = words.length;
  // Deterministic seed based on word characters
  const rawScores = [];
  const scaleFactor = isScaled ? 1 / Math.sqrt(dk) : 1.0;

  for (let i = 0; i < n; i++) {
    rawScores[i] = [];
    for (let j = 0; j < n; j++) {
      if (isMasked && j > i) {
        rawScores[i][j] = -1e9; // Causal mask
      } else {
        // High attention for syntactic relatives (like pronoun "its" attending to "law" and "application")
        let score = (Math.sin(i * 1.7 + j * 2.3) + Math.cos((i - j) * 1.1)) * 3;
        const wI = words[i].toLowerCase();
        const wJ = words[j].toLowerCase();
        if (wI === 'its' && (wJ === 'law' || wJ === 'application')) score += 8.0;
        if (i === j) score += 2.5; // diagonal bias
        rawScores[i][j] = score * scaleFactor;
      }
    }
  }

  // Softmax per row
  const softmaxWeights = [];
  for (let i = 0; i < n; i++) {
    softmaxWeights[i] = [];
    const maxVal = Math.max(...rawScores[i]);
    let sumExp = 0;
    const exps = [];
    for (let j = 0; j < n; j++) {
      const e = Math.exp(rawScores[i][j] - maxVal);
      exps.push(e);
      sumExp += e;
    }
    for (let j = 0; j < n; j++) {
      softmaxWeights[i][j] = sumExp > 0 ? (exps[j] / sumExp) : 0;
    }
  }

  // Render HTML grid
  let html = `
    <div style="margin-bottom:12px; font-size:0.8rem; color:var(--text-muted); text-align:center;">
      Hover over cells to see Query ($i$) → Key ($j$) attention distribution
    </div>
    <div style="display:flex; flex-direction:column; gap:4px;">
      <!-- Top column headers -->
      <div style="display:flex; margin-left:70px; gap:4px;">
  `;

  for (let j = 0; j < n; j++) {
    html += `<div class="heatmap-token-label" title="${esc(words[j])}">${esc(words[j])}</div>`;
  }
  html += `</div>`;

  // Rows
  for (let i = 0; i < n; i++) {
    html += `<div style="display:flex; align-items:center; gap:6px;">
      <div style="width:64px; font-size:0.72rem; color:var(--text-muted); text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${esc(words[i])}">
        ${esc(words[i])}
      </div>
      <div style="display:flex; gap:4px;">
    `;

    for (let j = 0; j < n; j++) {
      const w = softmaxWeights[i][j];
      const opacity = Math.min(1, Math.max(0.08, w * 1.5));
      const bg = isMasked && j > i
        ? '#f1f1f1'
        : `rgba(233, 162, 74, ${opacity})`;
      const textColor = '#1b1b1f';

      html += `
        <div class="heatmap-cell" style="background:${bg}; color:${textColor};"
          title="Q: '${esc(words[i])}' -> K: '${esc(words[j])}' | Weight: ${(w * 100).toFixed(1)}%">
          ${(w * 100).toFixed(0)}%
        </div>
      `;
    }
    html += `</div></div>`;
  }

  html += `</div>`;
  container.innerHTML = html;
}

// VIEW 3: Dimensions & Parameter Budget Studio
function renderDimensionStudioView() {
  const cfg = state.config;
  const d = computeDerivedParams(cfg);

  elements.stageContent.innerHTML = `
    <div class="dim-studio-wrapper">
      <!-- Interactive Inputs -->
      <div class="dim-inputs-card">
        <div class="dim-field">
          <label>Batch Size (B):</label>
          <input type="number" id="inp_b" min="1" max="256" value="${cfg.batch}">
        </div>

        <div class="dim-field">
          <label>Source Seq Length (n):</label>
          <input type="number" id="inp_n" min="1" max="4096" value="${cfg.n}">
        </div>

        <div class="dim-field">
          <label>Target Seq Length (m):</label>
          <input type="number" id="inp_m" min="1" max="4096" value="${cfg.m}">
        </div>

        <div class="dim-field">
          <label>Model Dimension (d_model):</label>
          <input type="number" id="inp_dmodel" min="32" max="8192" step="32" value="${cfg.d_model}">
        </div>

        <div class="dim-field">
          <label>Attention Heads (h):</label>
          <input type="number" id="inp_h" min="1" max="128" value="${cfg.h}">
        </div>

        <div class="dim-field">
          <label>FFN Inner Dimension (d_ff):</label>
          <input type="number" id="inp_dff" min="64" max="32768" value="${cfg.d_ff}">
        </div>

        <div class="dim-field">
          <label>Layers per Stack (N):</label>
          <input type="number" id="inp_n_layers" min="1" max="64" value="${cfg.N}">
        </div>

        <div class="dim-field">
          <label>Vocabulary Size (V):</label>
          <input type="number" id="inp_v" min="100" max="250000" value="${cfg.V}">
        </div>
      </div>

      <!-- Parameter Breakdown Table -->
      <div class="params-table-container">
        <table class="params-table">
          <thead>
            <tr>
              <th>Sublayer / Component</th>
              <th>Formula</th>
              <th>Count per Layer</th>
              <th>Total (N=${cfg.N})</th>
              <th>% of Model</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Shared Embeddings (Input, Output, Pre-Softmax)</td>
              <td class="mono">V × d_model</td>
              <td class="mono">-</td>
              <td class="mono highlight">${formatParamCount(d.shared_embedding_params)}</td>
              <td class="mono">${((d.shared_embedding_params / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Encoder Self-Attention (W_Q, W_K, W_V, W_O + biases)</td>
              <td class="mono">4 × d_model² + 4 × d_model</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params)}</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params * cfg.N)}</td>
              <td class="mono">${(((d.enc_self_attn_params * cfg.N) / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Encoder Feed-Forward Networks</td>
              <td class="mono">2 × d_model × d_ff + d_ff + d_model</td>
              <td class="mono">${formatParamCount(2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model)}</td>
              <td class="mono">${formatParamCount((2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model) * cfg.N)}</td>
              <td class="mono">${((((2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model) * cfg.N) / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Decoder Masked Self-Attention</td>
              <td class="mono">4 × d_model² + 4 × d_model</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params)}</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params * cfg.N)}</td>
              <td class="mono">${(((d.enc_self_attn_params * cfg.N) / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Decoder Cross-Attention (Encoder-Decoder)</td>
              <td class="mono">4 × d_model² + 4 × d_model</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params)}</td>
              <td class="mono">${formatParamCount(d.enc_self_attn_params * cfg.N)}</td>
              <td class="mono">${(((d.enc_self_attn_params * cfg.N) / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Decoder Feed-Forward Networks</td>
              <td class="mono">2 × d_model × d_ff + d_ff + d_model</td>
              <td class="mono">${formatParamCount(2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model)}</td>
              <td class="mono">${formatParamCount((2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model) * cfg.N)}</td>
              <td class="mono">${((((2 * cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_model) * cfg.N) / d.grand_total_params) * 100).toFixed(1)}%</td>
            </tr>
            <tr>
              <td>Layer Normalization Scales & Biases</td>
              <td class="mono">(2 enc + 3 dec) × 2 × d_model × N</td>
              <td class="mono">${formatParamCount(10 * cfg.d_model)}</td>
              <td class="mono">${formatParamCount(10 * cfg.d_model * cfg.N)}</td>
              <td class="mono">${(((10 * cfg.d_model * cfg.N) / d.grand_total_params) * 100).toFixed(2)}%</td>
            </tr>
            <tr style="background:#fdfdb3; font-weight:700;">
              <td>GRAND TOTAL MODEL PARAMETERS</td>
              <td class="mono">Sum of all sublayers</td>
              <td class="mono">-</td>
              <td class="mono highlight">${formatParamCount(d.grand_total_params)} (${d.grand_total_params.toLocaleString()})</td>
              <td class="mono">100.0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  // Attach live change listeners to inputs
  const inputs = ['inp_b', 'inp_n', 'inp_m', 'inp_dmodel', 'inp_h', 'inp_dff', 'inp_n_layers', 'inp_v'];
  inputs.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', (e) => {
        const val = parseInt(e.target.value);
        if (isNaN(val) || val <= 0) return;

        if (id === 'inp_b') cfg.batch = val;
        if (id === 'inp_n') cfg.n = val;
        if (id === 'inp_m') cfg.m = val;
        if (id === 'inp_dmodel') {
          cfg.d_model = val;
          cfg.d_ff = val * 4; // Auto-scale FFN
        }
        if (id === 'inp_h') cfg.h = val;
        if (id === 'inp_dff') cfg.d_ff = val;
        if (id === 'inp_n_layers') cfg.N = val;
        if (id === 'inp_v') cfg.V = val;

        state.presetKey = 'custom';
        elements.presetDropdown.value = 'custom';
        renderDimensionStudioView();
        updateLiveMetrics();
      });
    }
  });
}

// VIEW 4: Complexity Benchmark (Vaswani Table 1)
function renderBenchmarkView() {
  const n = state.config.n;
  const d = state.config.d_model;
  const k = 3; // Kernel size
  const r = 5; // Neighborhood

  elements.stageContent.innerHTML = `
    <div class="benchmark-wrapper">
      <div style="background:var(--bg-secondary); border:1px solid var(--border-subtle); border-radius:12px; padding:20px;">
        <h4 style="color:var(--accent-cyan); margin-bottom:8px; font-size:1.1rem;">
          Table 1 Reproduction: Architectural Complexity & Path Length
        </h4>
        <p style="color:var(--text-muted); font-size:0.88rem; margin-bottom:16px;">
          Comparing Self-Attention against Recurrent, Convolutional, and Restricted Self-Attention layers for sequence length $n = ${n}$ and representation dimension $d = ${d}$.
        </p>

        <table class="comp-table">
          <thead>
            <tr>
              <th>Layer Type</th>
              <th>Complexity per Layer</th>
              <th>Current FLOP Metric</th>
              <th>Sequential Operations</th>
              <th>Maximum Path Length</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td><strong>Self-Attention</strong> (Transformer)</td>
              <td>O(n² · d)</td>
              <td class="mono">${formatParamCount(n * n * d)} FLOPs</td>
              <td class="mono">O(1) [Fully Parallel]</td>
              <td class="mono highlight">O(1) [Direct Connection]</td>
            </tr>
            <tr>
              <td><strong>Recurrent</strong> (LSTM / GRU)</td>
              <td>O(n · d²)</td>
              <td class="mono">${formatParamCount(n * d * d)} FLOPs</td>
              <td class="mono" style="color:var(--accent-rose);">O(n) [Must wait t-1]</td>
              <td class="mono" style="color:var(--accent-rose);">O(n) [Long distance decay]</td>
            </tr>
            <tr>
              <td><strong>Convolutional</strong> (ByteNet / ConvS2S)</td>
              <td>O(k · n · d²)</td>
              <td class="mono">${formatParamCount(k * n * d * d)} FLOPs</td>
              <td class="mono">O(1)</td>
              <td class="mono">O(log_k(n))</td>
            </tr>
            <tr>
              <td><strong>Self-Attention (restricted)</strong></td>
              <td>O(r · n · d)</td>
              <td class="mono">${formatParamCount(r * n * d)} FLOPs</td>
              <td class="mono">O(1)</td>
              <td class="mono">O(n / r)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="callout emerald">
        <div class="callout-title">💡 Why O(1) Path Length Transformed AI</div>
        <p>In recurrent networks, gradients flowing from token $n$ back to token $1$ must multiply through $n$ transition weight matrices, causing exponential gradient decay or explosion. In Self-Attention, any word connects to any other word in <strong>exactly 1 step</strong>, allowing effortless long-distance contextual learning!</p>
      </div>
    </div>
  `;
}

// Global Event Listeners
function setupEventListeners() {
  // Preset selector
  elements.presetDropdown.addEventListener('change', (e) => {
    const key = e.target.value;
    if (PRESETS[key]) {
      state.presetKey = key;
      state.config = { ...PRESETS[key] };
      updateLiveMetrics();
      renderStageView();
    }
  });

  // Story Navigation
  elements.btnPrev.addEventListener('click', () => {
    if (state.currentChapter > 1) {
      loadChapter(state.currentChapter - 1);
    }
  });

  elements.btnNext.addEventListener('click', () => {
    if (state.currentChapter < state.totalChapters) {
      loadChapter(state.currentChapter + 1);
    }
  });

  // Tab View Switcher
  elements.tabBlueprint.addEventListener('click', () => switchView('blueprint'));
  elements.tabAttention.addEventListener('click', () => switchView('attention'));
  elements.tabDimensions.addEventListener('click', () => switchView('dimensions'));
  elements.tabBenchmark.addEventListener('click', () => switchView('benchmark'));

  // Close Drawer
  elements.btnCloseDrawer.addEventListener('click', () => {
    elements.inspectorDrawer.classList.remove('open');
  });

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight' || e.key === ']') {
      if (state.currentChapter < state.totalChapters) loadChapter(state.currentChapter + 1);
    } else if (e.key === 'ArrowLeft' || e.key === '[') {
      if (state.currentChapter > 1) loadChapter(state.currentChapter - 1);
    } else if (e.key === 'Escape') {
      elements.inspectorDrawer.classList.remove('open');
    }
  });
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', initApp);


const _renderStageView = renderStageView;
renderStageView = function () { if (window.Flow) Flow.stop(); _renderStageView.apply(this, arguments); renderMath(); };
renderBlueprintView = function () { Flow.mount(); };
