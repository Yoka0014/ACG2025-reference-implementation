
# ACG2025 Paper Reference Implementation

**Language:** [English](#english) | [日本語](#japanese)

## English

### 1. Overview

This project is a reference implementation of the method proposed in the paper "Mastering Othello with genetic algorithm and reinforcement learning". The implementation is based on a method that optimizes Othello evaluation functions using N-Tuple Systems combined with Biased Random-Key Genetic Algorithm (BRKGA) and Temporal-Difference (TD) learning.

Specifically, it consists of the following two main components:

1.  **N-Tuple System Structure Optimization**: Using BRKGA to search for effective n-tuple combinations (sets of board coordinates) as features for evaluation functions. Each n-tuple structure is evaluated by training an evaluation function with that structure using TD learning, and its performance determines the fitness.
2.  **Evaluation Function Parameter Learning**: Through self-play, TD(λ) method, a variant of TD learning, is used to learn the weights (parameters) of a given N-Tuple System.

Using this implementation, you can reproduce the experiments described in the paper and build your own evaluation functions.

### 2. Directory and File Structure

The roles of the main directories and files in this project are as follows:

-   `ACG2025-reference-implementation.sln`: Visual Studio solution file.
-   `ACG2025-reference-implementation/`: Directory containing the project source code.
    -   `Program.cs`: Main entry point. The functionality (tool) executed is switched by symbols defined at build time.
    -   `Engines/`: Game engines such as alpha-beta search and MCTS.
    -   `Evaluation/`: Classes related to evaluation functions.
    -   `Learn/`: Implementation of machine learning algorithms such as TD learning, self-play, and N-Tuple optimization using BRKGA.
    -   `NTupleSystem/`: Classes for N-Tuple System structure and management.
    -   `Reversi/`: Basic game logic including Othello rules, board representation, and move generation.
    -   `Search/`: Implementation of search algorithms (alpha-beta search, MCTS).
    -   `Protocols/`: Interfaces for communication between game engines and external GUIs, such as NBoard protocol.
    -   `Utils/`: Auxiliary utility classes.
-   `ConfigFileTemplates/`: Templates for configuration files (`.json`) used by each tool. Copy these files to any working directory for use.

### 3. Build Instructions

This project is developed with .NET 8. You can build it in an environment with .NET 8 SDK installed. Specify preprocessor symbols at build time according to the tool you want to use.

The following commands show how to build each tool:

- **N-Tuple System Structure Optimization (OPTIMIZE_NTUPLE)**
  ```sh
  dotnet build -c Release -p:DefineConstants=OPTIMIZE_NTUPLE ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

- **Evaluation Function Parameter Learning with TD Learning (RL_SELFPLAY)**
  ```sh
  dotnet build -c Release -p:DefineConstants=RL_SELFPLAY ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

- **Evaluation Function Generation from Pool (DECODE_POOL)**
  ```sh
  dotnet build -c Release -p:DefineConstants=DECODE_POOL ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

Upon successful build, an executable file `ACG2025-reference-implementation` with the specified functionality will be generated in the `ACG2025-reference-implementation/bin/Release/net8.0/` directory.

### 4. Tool Usage

This implementation includes three main tools corresponding to the experiments in the paper. These tools are enabled by specifying appropriate symbols in the build commands mentioned above.

#### 4.1. N-Tuple System Structure Optimization (OPTIMIZE_NTUPLE)

This corresponds to the "N-Tuple System Structure Optimization" experiment in the paper. It uses BRKGA to search for optimal n-tuple structures.

**Example Commands:**

```sh
# Start search from scratch
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    ntuple_optimizer_config.json \
    brkga_config.json \
    td_config.json \
    10 12 100

# Resume search from existing pool file
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    ntuple_optimizer_config.json \
    brkga_config.json \
    td_config.json \
    10 12 100 pool.bin
```

**Arguments:**
- `args[0]`: N-Tuple optimization overall configuration file (`ntuple_optimizer_config.json`)
- `args[1]`: BRKGA configuration file (`brkga_config.json`)
- `args[2]`: TD learning configuration file (`td_config.json`)
- `args[3]`: n-tuple size (e.g., `10`)
- `args[4]`: number of n-tuples (e.g., `12`)
- `args[5]`: number of BRKGA generations (e.g., `100`)
- `args[6]` (optional): pool file to resume search (`pool.bin`)

#### 4.2. Evaluation Function Parameter Learning with TD Learning (RL_SELFPLAY)

This corresponds to the "Evaluation Function Parameter Learning with TD Learning" experiment in the paper. It uses self-play and TD(λ) method to learn parameters (weights) of evaluation functions with specific n-tuple structures.

**Example Commands:**

```sh
# Continue learning with existing weights file
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    selfplay_config.json \
    value_func_weights.bin 10

# Start learning with zero-initialized weights
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    selfplay_config.json \
    value_func_weights.bin 10 zero
```

**Arguments:**
- `args[0]`: Self-play learning configuration file (`selfplay_config.json`)
- `args[1]`: Evaluation function weights file (`value_func_weights.bin`). An error occurs if it doesn't exist.
- `args[2]`: Number of learning cycles (e.g., `10`)
- `args[3]` (optional): Specify `zero` to initialize the weights file with zeros before starting learning.

#### 4.3. Evaluation Function Generation from Pool (DECODE_POOL)

This tool extracts top-performing individuals (n-tuple structures) from the pool file (`pool.bin`) generated by the `OPTIMIZE_NTUPLE` tool and generates corresponding evaluation function files (`.bin`) and n-tuple definition files (`.txt`).

**Example Commands:**

```sh
# Decode top 3 individuals from pool file and generate evaluation functions with 12 10-tuples
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    pool.bin 3 10 12
```

**Arguments:**
- `args[0]`: BRKGA pool file (`pool.bin`)
- `args[1]`: Number of top individuals to decode (e.g., `3`)
- `args[2]`: n-tuple size (e.g., `10`)
- `args[3]`: number of n-tuples (e.g., `12`)
- `args[4]` (optional): number of moves per phase in evaluation function (default: `60`)

Running this tool generates `value_func_weights_idv{i}.bin` and `ntuples_idv{i}.txt` in the current directory.

### 5. Configuration File Structure

The following shows the structure of configuration files used by each tool and parameter descriptions. Please refer to the template files in the `ConfigFileTemplates/` directory and adjust parameters as needed.

#### 5.1. N-Tuple Optimization Configuration File (`ntuple_optimizer_config.json`)

```json
{
  "NumTrainData": 10000,
  "NumTestData": 10000,
  "TrainDataVariationFactor": 0.05,
  "NumSimulations": 3200,
  "TrainDataUpdateInterval": 100
}
```

**Parameter Descriptions:**
- `NumTrainData`: Number of training data. The number of training game data used for evaluating each n-tuple structure.
- `NumTestData`: Number of test data. The number of test game data used for measuring evaluation function performance after training.
- `TrainDataVariationFactor`: Training data variation factor. A value from 0 to 1 that controls the diversity of training data.
- `NumSimulations`: Number of Monte Carlo simulations. The number of simulations used for search at each position.
- `TrainDataUpdateInterval`: Training data update interval. Updates training data every specified number of generations.

#### 5.2. BRKGA Configuration File (`brkga_config.json`)

```json
{
  "PopulationSize": 100,
  "EliteRate": 0.2,
  "MutantRate": 0.2,
  "EliteInheritanceProb": 0.7,
  "LearningRateForEval": 0.1,
  "NumEpochsForEval": 20,
  "PoolFileName": "pool",
  "FitnessHistoryFileName": "fitness_history"
}
```

**Parameter Descriptions:**
- `PopulationSize`: Population size. The number of individuals per generation in the genetic algorithm.
- `EliteRate`: Elite individual rate. The proportion of excellent individuals in the population (0 to 1).
- `MutantRate`: Mutant individual rate. The proportion of mutant individuals in the population (0 to 1).
- `EliteInheritanceProb`: Elite inheritance probability. The probability of inheriting elite individual genes during crossover.
- `LearningRateForEval`: Learning rate for evaluation. The learning rate of TD learning used when evaluating individuals.
- `NumEpochsForEval`: Number of epochs for evaluation. The number of TD learning epochs used for evaluating each individual.
- `PoolFileName`: Pool file name. Base name for pool files generated during the evolutionary process.
- `FitnessHistoryFileName`: Fitness history file name. File name for recording fitness progression of each generation.

#### 5.3. TD Learning Configuration File (`td_config.json`)

```json
{
  "NumEpisodes": 250000,
  "NumInitialRandomMoves": 1,
  "LearningRate": 0.2,
  "DiscountRate": 1,
  "InitialExplorationRate": 0.2,
  "FinalExplorationRate": 0.1,
  "EligibilityTraceFactor": 0.5,
  "HorizonCutFactor": 0.1,
  "TCLFactor": 2.7,
  "WeightsFileName": "value_func_weights_td",
  "SaveWeightsInterval": 10000,
  "SaveOnlyLatestWeights": true
}
```

**Parameter Descriptions:**
- `NumEpisodes`: Number of episodes. The number of games to run in TD learning.
- `NumInitialRandomMoves`: Number of initial random moves. The number of random moves at the start of each game.
- `LearningRate`: Learning rate. The learning rate for weight updates in TD learning (0 to 1).
- `DiscountRate`: Discount rate. The discount rate for future rewards (0 to 1).
- `InitialExplorationRate`: Initial exploration rate. The exploration rate of ε-greedy policy at the start of learning.
- `FinalExplorationRate`: Final exploration rate. The exploration rate of ε-greedy policy at the end of learning.
- `EligibilityTraceFactor`: Eligibility trace factor. The λ parameter of TD(λ) (0 to 1).
- `HorizonCutFactor`: Horizon cut factor. Parameter related to search termination.
- `TCLFactor`: TCL factor. Parameter related to Temporal Coherence Learning.
- `WeightsFileName`: Weights file name. Base name for files that save weights after learning.
- `SaveWeightsInterval`: Weights save interval. Saves weights every specified number of episodes.
- `SaveOnlyLatestWeights`: Save only latest weights. When true, keeps only the latest weights file.

#### 5.4. Self-Play Learning Configuration File (`selfplay_config.json`)

```json
{
  "NumThreads": 8,
  "NumSamplingMoves": 30,
  "NumSimulations": 800,
  "RootDirichletAlpha": 0.3,
  "RootExplorationFraction": 0.25,
  "NumGamesInBatch": 500000,
  "NumEpoch": 200,
  "StartWithRandomTrainData": true,
  "LearningRate": 1.0,
  "WeightsFileName": "value_func_weights_sp"
}
```

**Parameter Descriptions:**
- `NumThreads`: Number of threads. The number of threads used for parallel execution.
- `NumSamplingMoves`: Number of sampling moves. The number of moves to sample in each game.
- `NumSimulations`: Number of simulations. The number of simulations to run in MCTS.
- `RootDirichletAlpha`: Root Dirichlet alpha parameter. The α value of Dirichlet noise at the root node in MCTS search.
- `RootExplorationFraction`: Root exploration fraction. The proportion of Dirichlet noise used for exploration at the root node.
- `NumGamesInBatch`: Number of games in batch. The number of games included in one learning batch.
- `NumEpoch`: Number of epochs. The number of epochs to run in self-play learning.
- `StartWithRandomTrainData`: Start with random training data. When true, starts with random training data.
- `LearningRate`: Learning rate. The learning rate for neural network training.
- `WeightsFileName`: Weights file name. Base name for files that save weights after learning.

---

## Japanese

## 日本語

### 1. 概要

本プロジェクトは、論文「Mastering Othello with genetic algorithm and reinforcement learning」で提案された手法のリファレンス実装です。本実装は、N-Tuple Systemを用いたリバーシの評価関数を、Biased Random-Key Genetic Algorithm (BRKGA) と Temporal-Difference (TD) 学習を組み合わせて最適化する手法に基づいています。

具体的には、以下の2つの主要なコンポーネントから構成されています。

1.  **N-Tuple Systemの構造最適化**: BRKGAを用いて、評価関数の特徴として有効なn-tupleの組（盤上の座標の組）を探索します。個々のn-tuple構造の評価は、TD学習によってその構造を持つ評価関数の学習を行い、その性能によって決定されます。
2.  **評価関数のパラメータ学習**: 自己対戦を通じて、TD学習の一種であるTD(λ)法により、与えられたN-Tuple Systemの重み（パラメータ）を学習します。

本実装を用いることで、論文で述べられている実験を再現し、独自の評価関数を構築することが可能です。

### 2. ディレクトリ・ファイル構成

本プロジェクトの主要なディレクトリとファイルの役割は以下の通りです。

-   `ACG2025-reference-implementation.sln`: Visual Studio用のソリューションファイル。
-   `ACG2025-reference-implementation/`: プロジェクトのソースコードが含まれるディレクトリ。
    -   `Program.cs`: メインのエントリーポイント。ビルド時に定義するシンボルによって、実行される機能（ツール）が切り替わります。
    -   `Engines/`: α-β探索やMCTSなどの思考エンジン。
    -   `Evaluation/`: 評価関数に関連するクラス。
    -   `Learn/`: TD学習、自己対戦、BRKGAによるN-Tuple最適化など、機械学習アルゴリズムの実装。
    -   `NTupleSystem/`: N-Tuple Systemの構造や管理を行うクラス。
    -   `Reversi/`: リバーシのルール、盤面表現、着手生成など、ゲームの基本的なロジック。
    -   `Search/`: 探索アルゴリズム（α-β探索、MCTS）の実装。
    -   `Protocols/`: NBoardプロトコルなど、思考エンジンを外部のGUIと通信させるためのインターフェース。
    -   `Utils/`: 補助的なユーティリティクラス。
-   `ConfigFileTemplates/`: 各ツールで使用する設定ファイル（`.json`）のテンプレートが格納されています。これらのファイルを任意の作業ディレクトリにコピーして使用してください。

### 3. ビルド方法

本プロジェクトは .NET 8 で開発されています。.NET 8 SDKがインストールされている環境でビルドできます。使用したいツールに応じて、ビルド時にプリプロセッサシンボルを指定します。

以下に、各ツールをビルドするためのコマンドを示します。

- **N-Tuple Systemの構造最適化 (OPTIMIZE_NTUPLE)**
  ```sh
  dotnet build -c Release -p:DefineConstants=OPTIMIZE_NTUPLE ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

- **TD学習による評価関数パラメータの学習 (RL_SELFPLAY)**
  ```sh
  dotnet build -c Release -p:DefineConstants=RL_SELFPLAY ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

- **プールからの評価関数生成 (DECODE_POOL)**
  ```sh
  dotnet build -c Release -p:DefineConstants=DECODE_POOL ACG2025-reference-implementation/ACG2025-reference-implementation.csproj
  ```

ビルドが成功すると、`ACG2025-reference-implementation/bin/Release/net8.0/` ディレクトリに、指定した機能を持つ実行ファイル `ACG2025-reference-implementation` が生成されます。

### 4. ツールの使用方法

本実装には、論文の実験に対応する3つの主要なツールが含まれています。これらのツールは、前述のビルドコマンドで適切なシンボルを指定することで有効になります。

#### 4.1. N-Tuple Systemの構造最適化 (OPTIMIZE_NTUPLE)

これは論文の「N-Tuple Systemの構造最適化」実験に対応します。BRKGAを用いて最適なn-tupleの構造を探索します。

**実行コマンド例:**

```sh
# ゼロから探索を開始する場合
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    ntuple_optimizer_config.json \
    brkga_config.json \
    td_config.json \
    10 12 100

# 既存のプールファイルから探索を再開する場合
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    ntuple_optimizer_config.json \
    brkga_config.json \
    td_config.json \
    10 12 100 pool.bin
```

**引数:**
- `args[0]`: N-Tuple最適化の全体設定ファイル (`ntuple_optimizer_config.json`)
- `args[1]`: BRKGAの設定ファイル (`brkga_config.json`)
- `args[2]`: TD学習の設定ファイル (`td_config.json`)
- `args[3]`: n-tupleのサイズ (例: `10`)
- `args[4]`: n-tupleの数 (例: `12`)
- `args[5]`: BRKGAの世代数 (例: `100`)
- `args[6]` (オプション): 探索を再開するためのプールファイル (`pool.bin`)

#### 4.2. TD学習による評価関数パラメータの学習 (RL_SELFPLAY)

これは論文の「TD学習による評価関数のパラメータ学習」実験に対応します。自己対戦とTD(λ)法を用いて、特定のn-tuple構造を持つ評価関数のパラメータ（重み）を学習します。

**実行コマンド例:**

```sh
# 既存の重みファイルを用いて学習を継続する場合
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    selfplay_config.json \
    value_func_weights.bin 10

# 重みをゼロで初期化して学習を開始する場合
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    selfplay_config.json \
    value_func_weights.bin 10 zero
```

**引数:**
- `args[0]`: 自己対戦学習の設定ファイル (`selfplay_config.json`)
- `args[1]`: 評価関数の重みファイル (`value_func_weights.bin`)。存在しない場合はエラーになります。
- `args[2]`: 学習サイクル数 (例: `10`)
- `args[3]` (オプション): `zero` を指定すると、重みファイルをゼロで初期化してから学習を開始します。

#### 4.3. プールからの評価関数生成 (DECODE_POOL)

`OPTIMIZE_NTUPLE` ツールによって生成されたプールファイル (`pool.bin`) から、成績上位の個体（n-tuple構造）を取り出し、それに対応する評価関数ファイル (`.bin`) とn-tuple定義ファイル (`.txt`) を生成します。

**実行コマンド例:**

```sh
# プールファイルから上位3個体をデコードし、10-tupleが12個の評価関数を生成
./ACG2025-reference-implementation/bin/Release/net8.0/ACG2025-reference-implementation \
    pool.bin 3 10 12
```

**引数:**
- `args[0]`: BRKGAのプールファイル (`pool.bin`)
- `args[1]`: デコードする上位個体の数 (例: `3`)
- `args[2]`: n-tupleのサイズ (例: `10`)
- `args[3]`: n-tupleの数 (例: `12`)
- `args[4]` (オプション): 評価関数の1フェーズあたりの手数 (デフォルト: `60`)

このツールを実行すると、カレントディレクトリに `value_func_weights_idv{i}.bin` と `ntuples_idv{i}.txt` が生成されます。

### 5. 設定ファイルの構造

各ツールで使用する設定ファイルの構造とパラメータの説明を以下に示します。`ConfigFileTemplates/` ディレクトリにあるテンプレートファイルを参考に、必要に応じてパラメータを調整してください。

#### 5.1. N-Tuple最適化設定ファイル (`ntuple_optimizer_config.json`)

```json
{
  "NumTrainData": 10000,
  "NumTestData": 10000,
  "TrainDataVariationFactor": 0.05,
  "NumSimulations": 3200,
  "TrainDataUpdateInterval": 100
}
```

**パラメータ説明:**
- `NumTrainData`: 学習用データ数。各n-tuple構造の評価に使用する学習用ゲームデータの数。
- `NumTestData`: テスト用データ数。学習後の評価関数の性能測定に使用するテスト用ゲームデータの数。
- `TrainDataVariationFactor`: 学習データのバリエーション係数。0から1の値で、学習データの多様性を制御。
- `NumSimulations`: モンテカルロシミュレーション数。各局面での探索に使用するシミュレーション数。
- `TrainDataUpdateInterval`: 学習データ更新間隔。指定した世代数ごとに学習用データを更新。

#### 5.2. BRKGA設定ファイル (`brkga_config.json`)

```json
{
  "PopulationSize": 100,
  "EliteRate": 0.2,
  "MutantRate": 0.2,
  "EliteInheritanceProb": 0.7,
  "LearningRateForEval": 0.1,
  "NumEpochsForEval": 20,
  "PoolFileName": "pool",
  "FitnessHistoryFileName": "fitness_history"
}
```

**パラメータ説明:**
- `PopulationSize`: 集団サイズ。遺伝的アルゴリズムの1世代あたりの個体数。
- `EliteRate`: エリート個体率。集団における優秀な個体の比率（0から1）。
- `MutantRate`: 突然変異個体率。集団における突然変異個体の比率（0から1）。
- `EliteInheritanceProb`: エリート継承確率。交叉時にエリート個体の遺伝子を継承する確率。
- `LearningRateForEval`: 評価用学習率。個体の評価時に使用するTD学習の学習率。
- `NumEpochsForEval`: 評価用エポック数。各個体の評価に使用するTD学習のエポック数。
- `PoolFileName`: プールファイル名。進化過程で生成されるプールファイルのベース名。
- `FitnessHistoryFileName`: 適応度履歴ファイル名。各世代の適応度の推移を記録するファイル名。

#### 5.3. TD学習設定ファイル (`td_config.json`)

```json
{
  "NumEpisodes": 250000,
  "NumInitialRandomMoves": 1,
  "LearningRate": 0.2,
  "DiscountRate": 1,
  "InitialExplorationRate": 0.2,
  "FinalExplorationRate": 0.1,
  "EligibilityTraceFactor": 0.5,
  "HorizonCutFactor": 0.1,
  "TCLFactor": 2.7,
  "WeightsFileName": "value_func_weights_td",
  "SaveWeightsInterval": 10000,
  "SaveOnlyLatestWeights": true
}
```

**パラメータ説明:**
- `NumEpisodes`: エピソード数。TD学習で実行するゲーム数。
- `NumInitialRandomMoves`: 初期ランダム手数。ゲーム開始時のランダムな着手数。
- `LearningRate`: 学習率。TD学習の重み更新における学習率（0から1）。
- `DiscountRate`: 割引率。将来の報酬に対する割引率（0から1）。
- `InitialExplorationRate`: 初期探索率。学習開始時のε-greedyポリシーの探索率。
- `FinalExplorationRate`: 最終探索率。学習終了時のε-greedyポリシーの探索率。
- `EligibilityTraceFactor`: 適格度トレース係数。TD(λ)のλパラメータ（0から1）。
- `HorizonCutFactor`: ホライズンカット係数。探索の打ち切りに関するパラメータ。
- `TCLFactor`: TCL係数。Temporal Coherence Learning関連のパラメータ。
- `WeightsFileName`: 重みファイル名。学習後の重みを保存するファイルのベース名。
- `SaveWeightsInterval`: 重み保存間隔。指定したエピソード数ごとに重みを保存。
- `SaveOnlyLatestWeights`: 最新重みのみ保存。trueの場合、最新の重みファイルのみを保持。

#### 5.4. 自己対戦学習設定ファイル (`selfplay_config.json`)

```json
{
  "NumThreads": 8,
  "NumSamplingMoves": 30,
  "NumSimulations": 800,
  "RootDirichletAlpha": 0.3,
  "RootExplorationFraction": 0.25,
  "NumGamesInBatch": 500000,
  "NumEpoch": 200,
  "StartWithRandomTrainData": true,
  "LearningRate": 1.0,
  "WeightsFileName": "value_func_weights_sp"
}
```

**パラメータ説明:**
- `NumThreads`: スレッド数。並列実行に使用するスレッド数。
- `NumSamplingMoves`: サンプリング手数。各ゲームでサンプリングする着手数。
- `NumSimulations`: シミュレーション数。MCTSで実行するシミュレーション数。
- `RootDirichletAlpha`: ルートディリクレαパラメータ。MCTS探索のルートノードにおけるディリクレノイズのα値。
- `RootExplorationFraction`: ルート探索率。ルートノードでの探索に使用するディリクレノイズの比率。
- `NumGamesInBatch`: バッチ内ゲーム数。1つの学習バッチに含まれるゲーム数。
- `NumEpoch`: エポック数。自己対戦学習で実行するエポック数。
- `StartWithRandomTrainData`: ランダム学習データ開始。trueの場合、ランダムな学習データから開始。
- `LearningRate`: 学習率。ニューラルネットワークの学習率。
- `WeightsFileName`: 重みファイル名。学習後の重みを保存するファイルのベース名。
