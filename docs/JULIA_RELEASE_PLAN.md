# Julia 版 mappy のリリース計画

対象: `mappy-ds` 0.3.0（Python，`src/mappy_ds`）に対応する Julia パッケージを
General レジストリに登録し，Python 版と同じドキュメントサイト
（`Yuu-Miino/doc.mappy`）で公開するまでの計画。2026-11-08 時点の Julia
エコシステム（Julia 1.13.3 安定版 / 1.10.13 LTS，OrdinaryDiffEq v7，
Documenter 1.20）を前提にしている。

**決定事項（2026-11-08 ヒアリング済み）**

| 項目 | 決定 |
| --- | --- |
| パッケージ名 | `MappyDS.jl`（モジュール `MappyDS`，リポジトリ `Yuu-Miino/MappyDS.jl`，既定ブランチ `main`） |
| docs 配備先 | `doc.mappy/julia/`（Documenter `deploy_repo` + `dirname`） |
| `Mode.step` の Julia 名 | `evolve` |
| `param_keys` | パラメータ NamedTuple の全キー |
| 初回登録 | 0.1.0 完成時に General へ登録 |
| Julia compat | `julia = "1.10"`（LTS） |
| フォーマッタ | Runic |
| 実装体制 | Claude のセッションで実装し，人手は §11 のリポジトリ作成・secrets・登録コメントに限る |

**要約**

- パッケージ名は **`MappyDS.jl`**（モジュール `MappyDS`）。`Mappy`
  は General の名前類似チェック（既存の `Snappy`，`MRphy` と編集距離 2）に
  掛かる見込みで，`MappyDS` は通過する。PyPI 名 `mappy-ds` とも揃う。
- 導関数は Python 版と同じ**解析的な変分方程式＋境界補正**を移植する。
  ユーザ関数の f, Df, D²f, ∂ₚf, ∂ₚDf は SymPy トレースの代わりに
  **ForwardDiff.jl**（StaticArrays 上の out-of-place 関数）で得る。
  Symbolics.jl は任意の拡張に留める。
- ODE は **OrdinaryDiffEq v7 のサブライブラリ**（`OrdinaryDiffEqTsit5`，
  `OrdinaryDiffEqVerner`）と境界ごとの `ContinuousCallback` で解く。
- 継続法・分岐追跡は自前実装を維持する（BifurcationKit.jl の写像対応は
  「limited」と自己申告されており，混合微分や θ パラメータ化の分岐条件が
  載らない）。BifurcationKit 連携は 1.0 以降の拡張候補。
- **0.1.0 → 0.2.0 → 0.3.0** の 3 段階で Python 版 0.1.0 / 0.2.0 / 0.3.0 の
  機能に追いつき，その後 1.0.0 で API を凍結する。
- docs は Documenter の `deploy_repo` + `dirname="julia"` で
  `https://yuu-miino.github.io/doc.mappy/julia/` に配備する。既存の Sphinx
  サイト（`v0.3.0/`，`latest` 等）とは衝突しない。
- 登録は Registrator（`@JuliaRegistrator register`）で行い，新規パッケージは
  **3 日間**の待機後に自動マージされる。TagBot・Dependabot（Julia 対応済み，
  CompatHelper は保守モード）を併用する。

---

## 1. ゴールと非ゴール

**ゴール**

- Python 版 0.3.0 と数値的に一致する（参照値で相対 1e-6 以内）Julia 版を
  General レジストリに登録する。
- モード概念（Miino, NOLTA 2024）の API をそのまま Julia 流に写す。
  概念名（mode / border / transition / Poincaré map / `image_detail`）は保つ。
- 導関数（Jacobian・Hessian・パラメータ微分・混合 2 階微分）は解析的に
  求め，AD（自動微分）はユーザ関数の微分とテストの検証にだけ使う。
- Python 版と Julia 版を同じ docs サイトに置き，相互にリンクする。

**非ゴール（1.0 まで扱わない）**

- Python 版からの自動移行ツール，両言語をまたぐランタイム連携
  （PythonCall を使ったクロス検証テストは別，§7.3）。
- BifurcationKit.jl / ModelingToolkit.jl との統合。
- GPU，疎行列，高次元（dim > 10 程度）向けの最適化。

## 2. 現状の棚卸し（Python 版 0.3.0）

移植対象の機能と規模。ソース約 3,900 行，テスト 193 件。

| モジュール | 公開 API | 行数 | 実装上の要点 |
| --- | --- | ---: | --- |
| `functions` | `SystemFunction`, `@vector_field`, `@mapping`, `@border` | 428 | SymPy でトレースして f, Df, D²f, ∂ₚf, ∂ₚDf を `lambdify`；`p[key]` の読み取りからパラメータ名を発見 |
| `modes` | `Mode`, `ContinuousMode`, `DiscreteMode`, `StepResult` | 539 | `solve_ivp` + terminal events，変分方程式 4 種（J, P, M, H），境界補正 |
| `maps` | `Map`, `PoincareMap`, `Diffeomorphism`, `MapResult`, `Transitions` | 634 | 遷移規則の 3 形式の正規化とグラフ検証，連鎖律 `_compose` |
| `continuation` | `continuation`, `arclength_continuation`, `ContinuationStep`, `ArclengthPoint`, `StepFunctionResult` | 435 | 自然パラメータ掃引，擬弧長法（Keller 1977；適応ステップ，fold 通過，閉曲線検出） |
| `root` | `find_cycle`, `trace_cycle`, `trace_cycle_arclength`, `find_local_bf`, `trace_local_bf`, `trace_local_bf_arclength`, `CycleResult`, `LocalBfResult` | 1,073 | `scipy.optimize.root` に解析ヤコビアン，余因子行列による完全解析な分岐条件のヤコビアン，共鳴フラグ |
| `plot` | `plot2d`, `plot_trajectory`, `PlotConfig`, `LineStyle` | 465 | matplotlib 対話ビューア（extra `plot`） |
| `trajectory`, `exceptions`, `_typing` | `Trajectory`, `TrajectorySegment`, `MappyError` 系 5 種 | 198 | |

そのまま再利用できる検証資産:

- `tests/conftest.py` の 4 系（Izhikevich，Hénon，強制 Duffing，閉形式の
  Poincaré 写像を持つ区分線形リレー振動子）と有限差分ヘルパ。
- 論文の参照値。例: Izhikevich（I = 10）で y = −3.4783849356，
  jac = 0.27080365164，hes = −0.193258205836；Duffing の周期倍分岐
  B ≈ 0.171687，0.368655；Alpazur の B1 ≈ 9.0777。
- `examples/` 5 本（論文 Fig. 8，Fig. 10 の再現を含む）。

## 3. パッケージ名とリポジトリ

### 3.1 名前の候補と AutoMerge の名前規則

General レジストリの AutoMerge は新規パッケージ名に次を要求する
（RegistryCI.jl `AutoMerge/src/guidelines.jl`）。

- 先頭大文字・ASCII 英数字のみ，**5 文字以上**，`julia` を含まない，`Ju` で
  始まらない，`jl` で終わらない。
- 既存名との **Damerau–Levenshtein 距離 ≥ 3**（小文字化して ≥ 2），かつ
  DL 距離 ≤ 4 の相手には視覚距離 > 2.5。
- リポジトリ URL は `/<Name>.jl.git` で終わること。

2026-11-08 時点の `Registry.toml`（14,706 件）に対する確認結果:

| 候補 | 判定 | 根拠 |
| --- | --- | --- |
| `Mappy` | **不可の見込み** | `Snappy`，`MRphy` と DL 距離 2 |
| `MapPy` | 不可 | 小文字化で `mrphy` と距離 2 |
| `ModeMaps` | 不可 | `MoreMaps` と距離 1 |
| **`MappyDS`** | **通過（推奨）** | 最近傍 `MRphy`，`Maybe`，`Snappy` が距離 3 以上。視覚距離は未計算 |
| `PoincareMaps` | 要注意 | `PlanarMaps`，`LinearMaps`，`PoincareDisk` が DL ≤ 4 で視覚距離チェック対象 |
| `HybridPoincare`，`ModeConcept` | 通過 | DL 距離 4 以内の既存名なし |
| `HybridSystems` | 使用済み | blegat 氏のハイブリッドオートマトン基盤 |

**決定: `MappyDS.jl`**。理由は (1) PyPI 名 `mappy-ds`・docs サイト
`doc.mappy` との連続性，(2) 距離チェック通過，(3) 一般名詞
（`PoincareMaps`）を占有しないこと。登録前に
`AutoMerge.meets_distance_check("MappyDS", all_pkg_names)` で視覚距離まで
ローカル確認する。万一掛かった場合の代替は `HybridPoincare.jl`。
類似が指摘された場合は triage 権限者がラベル
`Override AutoMerge: name similarity is okay` で通すこともできる。

### 3.2 リポジトリ構成

- 新規リポジトリ `Yuu-Miino/MappyDS.jl`（既定ブランチ `main`）。Python の
  `mappy` リポジトリとは分ける（Pkg の subdir 登録は可能だが，TagBot の
  `tag_prefix` や Documenter の設定が複雑になり，Python 側の `v*` タグと
  衝突する）。
- 雛形は PkgTemplates.jl 0.7 で生成する。プラグイン:
  `Git`, `License(MIT)`, `Tests(aqua=true, jet=true, project=true)`,
  `GitHubActions`, `Codecov`, `Documenter{GitHubActions}`, `TagBot`,
  `Dependabot`, `Runic`, `Citation`, `ColPracBadge`。CompatHelper は保守モードなので使わない。
- ライセンスは MIT を踏襲。

```
MappyDS.jl/
├── Project.toml            # name, uuid (uuid4), version, [deps], [weakdeps], [extensions], [compat]
├── src/
│   ├── MappyDS.jl          # module, exports, PrecompileTools workload
│   ├── functions.jl        # SystemFunction 相当（ForwardDiff）
│   ├── modes.jl            # ContinuousMode / DiscreteMode / StepResult
│   ├── maps.jl             # PoincareMap / Diffeomorphism / MapResult
│   ├── continuation.jl
│   ├── root/cycle.jl, root/local_bf.jl
│   ├── trajectory.jl, exceptions.jl
│   └── plot_api.jl         # plot2d / plot_trajectory のスタブ（本体は拡張）
├── ext/MappyDSMakieExt.jl  # weakdep: Makie
├── test/                   # Project.toml, runtests.jl, *_test.jl, fixtures/reference.json
├── docs/                   # make.jl, Project.toml, src/, Literate 用 examples
├── examples/               # Python examples/ と同名の 5 本（Literate 形式）
├── CHANGELOG.md, CITATION.bib, README.md, LICENSE
└── .github/workflows/      # CI.yml, TagBot.yml, Documentation.yml; dependabot.yml
```

## 4. 設計方針

### 4.1 API 対応表

Python の「デコレータで宣言 → クラスに渡す → `image_detail`」を，Julia では
「関数をそのまま渡す → 構造体を作る → `image_detail`」にする。キーワード
引数名は Python と揃え，モード名・境界名・パラメータ名は `Symbol` にする。

| Python (`mappy_ds`) | Julia (`MappyDS`) | 備考 |
| --- | --- | --- |
| `@vector_field(dim) def f(y, p)` | `VectorField(f, dim)` | `f(y, p)` は out-of-place，`y::SVector`，`p::NamedTuple` |
| `@mapping(dom, cod)` | `Mapping(f, dom, cod=dom)` | |
| `@border(dim, direction, name)` | `Border(q, dim; direction=1, name=nameof(q))` | |
| `SystemFunction.jacobian / hessian / parameter_jacobian / mixed_hessian` | `jacobian(f, y, p)`, `hessian(f, y, p)`, `parameter_jacobian(f, y, p)`, `mixed_hessian(f, y, p)` | 多重ディスパッチの関数 |
| `param_keys`（トレースで発見） | `param_keys(map) = keys(p)` | §4.2。発見ではなく NamedTuple の全キー |
| `ContinuousMode(name, fun, borders=..., max_interval, method, rtol, atol)` | `ContinuousMode(name, f; borders=(), max_interval=20.0, alg=Tsit5(), reltol=1e-6, abstol=1e-9, solver_kwargs...)` | `alg` は OrdinaryDiffEq の任意のアルゴリズム |
| `DiscreteMode(name, fun)` | `DiscreteMode(name, f)` | |
| `Mode.step(y0, params; jacobian, hessian, trajectory, param_keys, mixed_hessian)` | `evolve(mode, y0, p; jacobian, hessian, trajectory, param_keys, mixed_hessian) -> StepResult` | `Base.step` との衝突を避けて `evolve`（決定） |
| `PoincareMap(modes, transitions, initial_mode; end_modes, max_transitions)` | 同名 | `transitions::Dict{Symbol, Union{Symbol, Vector{Symbol}, Dict{Symbol,Symbol}}}` |
| `Diffeomorphism(fun)` | 同名 | |
| `Map.image / image_detail / trajectory` | `image`, `image_detail`, `trajectory` | `MapResult` の `jac`, `hes`, `jac_p`, `hes_p` は `nothing` 許容 |
| `MapResult.eigvals / eigvecs` | `eigvals(res)`, `eigvecs(res)` | `LinearAlgebra` の関数を拡張 |
| `root.find_cycle(diffeo, y0, params; period, **root_options)` | `find_cycle(map, y0, p; period=1, alg=SimpleTrustRegion(), kwargs...)` | §4.5 |
| `root.find_local_bf(..., param_key, theta)` | `find_local_bf(map, y0, p, :I, θ; period=1)` | |
| `root.trace_cycle / trace_local_bf` | 同名 | 自然パラメータ継続 |
| `root.trace_cycle_arclength / trace_local_bf_arclength` | 同名 | イベントは `Vector{Symbol}`（`:fold`, `:branch_point`, `:period_doubling`, `:neimark_sacker`, `:resonance_1_3`, …） |
| `continuation.continuation / arclength_continuation` | 同名 | ユーザ関数は `u -> (F, J)` のまま |
| `plot.plot2d / plot_trajectory / PlotConfig / LineStyle` | 同名，実体は `MappyDSMakieExt` | §4.6 |
| `MappyError` 系 | `MappyError <: Exception` と `UnknownModeError`, `TransitionError`, `ModeNotTerminatedError`, `DerivativeError` | `SymbolicEvaluationError` は AD 失敗の `DerivativeError` に読み替え |
| `CycleResult.success / message` | `retcode::SciMLBase.ReturnCode.T` | `successful_retcode` で判定できる |

### 4.2 ユーザ関数の導関数（SymPy トレースの置き換え）

**ForwardDiff.jl（1.4）を直接使う。** ユーザ関数は
`f(y::SVector{n}, p::NamedTuple) -> SVector{m}` と書いてもらい，

- Df: `ForwardDiff.jacobian(y -> f(y, p), y)`
- D²f: `ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(…)), y)` を
  `m × n × n` に整形（ベクトル値関数の Hessian は ForwardDiff の公式手順）
- ∂ₚf，∂ₚDf: `p` を `Tuple`/`SVector` に平坦化して `(y, p)` を連結した
  入力で微分し，列を切り出す

で得る。SymPy トレースと比べた利点は，(1) `if y[1] > 0` のような値依存の
分岐や任意の Julia コードがそのまま使える，(2) 「`sympy.cos` を使え」という
制約が消える，(3) 依存が軽い。欠点は Hessian が入れ子 Dual になることだが，
n ≤ 10 の StaticArrays 上ではアロケーションなしで十分速い。

Python 版が `p[key]` の読み取りから `param_keys` を発見していたのに対し，
Julia 版は**パラメータ NamedTuple の全キー**を `param_keys` とする（決定。
使っていないパラメータの列は自動的に 0）。`eval_parameter_jacobian_for`
の列合わせロジックが不要になる。

Symbolics.jl（7.x）による式生成は，コード生成を望むユーザ向けに
`MappyDSSymbolicsExt` として 0.4 以降に検討する。トレース制約・
RuntimeGeneratedFunctions の world-age 配慮・読み込み時間の重さが
あるので既定にはしない。ModelingToolkit v11 は構造簡約部分が AGPL に
なったため依存しない。

### 4.3 連続モード：ODE 積分とイベント

- 依存は **OrdinaryDiffEq v7 のサブライブラリ**に限定する:
  `OrdinaryDiffEqTsit5`（既定 `Tsit5()`，scipy `RK45` 相当）と
  `OrdinaryDiffEqVerner`（`Vern7()`/`Vern9()`，高精度用）。`SciMLBase`
  （`ODEProblem`，`solve`，`ContinuousCallback`，`terminate!`）は
  `OrdinaryDiffEqTsit5` が再エクスポートする。`DifferentialEquations.jl`
  全体には依存しない。
- 境界は **`ContinuousCallback` を 1 境界 1 個**作って `CallbackSet` に
  まとめる。方向は `affect!`（負→正）と `affect_neg!`（正→負）の片方を
  `nothing` にして表現し，クロージャで「どの境界が発火したか」を記録して
  `terminate!` する。`VectorContinuousCallback` は v7 で `affect!` の
  シグネチャが変わり，初期リリースで符号が反転していた経緯があるので
  使わない。
- 停止判定: `sol.retcode == ReturnCode.Terminated` なら境界到達，
  `ReturnCode.Success`（`tspan` 終端 = `max_interval`）で境界があるなら
  `ModeNotTerminatedError`。
- イベント時刻は内部で `IntervalNonlinearProblem` を浮動小数点の分解能まで
  解くので，scipy の既定より高精度になる（参照値の比較では Python 側の
  `rtol=1e-6` が支配的）。

### 4.4 変分方程式と境界補正

Python 版の `ContinuousMode._rhs` / `_border_correction` を**式ごと**
移植する。理由:

- Hessian と混合 2 階微分まで必要で，AD を ODE ソルバに二重に通すのは
  コスト（n² 本の Dual レーン）と信頼性の両面で不利。
- ForwardDiff を `solve` に通した場合，到達時刻の微分は 2026 年 2 月以降
  自動で伝播するようになったが，∂q/∂t が `sqrt(eps)` 幅の割線（約 1e-8
  相対精度）で，解析的な `∇q·f` ではない。入れ子 Dual（Hessian）の挙動は
  未検証。
- 境界補正の式（`B = I − f⊗∇q/(∇q·f)`，`∂τ/∂y0 = −(∇q J)/(∇q·f)`）は
  論文で導出済みで，有限差分と AD の両方で検証できる。

実装: 拡大状態 `z = (y, J, P, M, H)` を通常の `Vector{Float64}` に置き，
in-place の右辺 `rhs!(dz, z, p, t)` で `view` を切って書く。`einsum` は
素朴な多重ループで十分（n ≤ 10）。D²f の評価は §4.2 の入れ子 ForwardDiff。
ForwardDiff-through-`solve` と有限差分は**テストのオラクル**としてだけ使う。

### 4.5 周期点・分岐・継続

- `find_cycle`: `SciMLBase.NonlinearProblem(NonlinearFunction(F; jac=J), y0, p)`
  を `SimpleNonlinearSolve.jl`（2.x，軽量）の `SimpleTrustRegion()`
  （scipy `hybr` の Powell dogleg に近い）で解く。問題型は NonlinearSolve.jl
  と共通なので，ユーザが `NonlinearSolve` を読み込めば `NewtonRaphson()`
  や `TrustRegion()` を `alg=` に渡せる。
- `find_local_bf`: 余因子行列を使う完全解析ヤコビアン（`_cofactor`，
  `_bifurcation_system`）をそのまま移植。`det` は `LinearAlgebra.det`。
- `arclength_continuation` / `trace_*_arclength`: 自前実装を維持する。
  BifurcationKit.jl 0.8 は写像の不動点継続と PD / NS の標準形計算には
  対応するが，自己申告で「(limited) capabilities related to maps」であり，
  PD / NS **曲線**の追跡（`PDMAProblem` 等）は流れの周期軌道向け，
  θ パラメータ化した `det(DTⁿ − e^{iθ}I)` 形式や共鳴フラグは持たない。
  1.0 以降に `T^n(y) − y` を `BifurcationProblem`（`J`, `d2F`, `R01`, `R11`
  を供給）として包む拡張を検討する。

### 4.6 プロット

- Julia ≥ 1.9 の**パッケージ拡張**で `[weakdeps] Makie`，
  `[extensions] MappyDSMakieExt = "Makie"` とし，本体にはスタブ関数と
  `PlotConfig` / `LineStyle` だけを置く（Makie 0.24 で `MakieCore` が
  廃止されたので拡張は `Makie` 本体に対して書く。BifurcationKit と同じ構成）。
- `plot2d` は `Observable` と `on(events(ax.scene).mousebutton)` で
  クリック再開・`p` キーでパラメータ切替・上下キーで増減を実装する。
  GLMakie（ネイティブ窓）が matplotlib 対話ビューアの対応物，WGLMakie は
  ノートブック向け。
- 静的な `plot_trajectory` は同じ拡張に入れる。Plots.jl 向け `RecipesBase`
  レシピは要望があれば 0.4 以降。

### 4.7 依存関係と互換性

| 区分 | パッケージ（2026-11 時点の版） | 役割 |
| --- | --- | --- |
| 必須 | `ForwardDiff` 1.4, `StaticArrays` 1.9 | ユーザ関数の導関数 |
| 必須 | `OrdinaryDiffEqTsit5` 2.1, `OrdinaryDiffEqVerner` 2.4, `SciMLBase` 3.58 | ODE とイベント |
| 必須 | `SimpleNonlinearSolve` 2.14 | 周期点の Newton 法 |
| 必須 | `PrecompileTools` 1.3 | 読み込み後の初回実行短縮 |
| stdlib | `LinearAlgebra`, `Printf` | |
| weak | `Makie` 0.24 | プロット拡張 |
| test | `Test`, `Aqua` 0.8, `JET` 0.13, `JSON3`（参照値），`PythonCall` 0.9 + `CondaPkg`（任意） | |
| docs | `Documenter` 1.20, `DocumenterInterLinks`, `DocumenterCitations`, `Literate` 2.21 | |

- `julia = "1.10"`（LTS，決定）。AutoMerge は compat の最小・最大 minor
  （1.10 と 1.13）で `import` を試すので両方で CI を回す。
- `[compat]` は**全ての非 stdlib 依存に上限付き**で書く（AutoMerge 必須）。
  OrdinaryDiffEq 系は v7 の `VectorContinuousCallback` 符号修正後の版を下限に
  する（§4.3 の方針なら直接は影響しないが，念のため）。
- `--trim`（Julia 1.12 以降，実験的）はライブラリには関係しない。

## 5. バージョン計画とマイルストーン

Python 版のバージョン番号とは**同期させない**（Pkg の 0.x は minor が
破壊的変更の単位で，Python 側の都合で上げると互換性の意味が壊れる）。
代わりに docs に機能対応表を置く。

| Julia | 内容 | Python 版での対応 | 完了条件 |
| --- | --- | --- | --- |
| **0.1.0** | `VectorField` / `Mapping` / `Border`，`ContinuousMode` / `DiscreteMode`，`PoincareMap` / `Diffeomorphism`，Jacobian・Hessian（変分方程式＋境界補正），`find_cycle`，`continuation` / `trace_cycle`，`Trajectory`，例外 | 0.1.0 | Izhikevich / Hénon / Duffing / リレー振動子の参照値が 1e-6 で一致，有限差分・ForwardDiff オラクル検証，Aqua・JET・doctest 通過，docs が `doc.mappy/julia/` に出る，General 登録完了 |
| **0.2.0** | `jac_p`（パラメータ微分），`arclength_continuation`，`trace_cycle_arclength`（`:fold` / `:branch_point` / `:period_doubling` / `:neimark_sacker`） | 0.2.0 | Python 版 `test_continuation.py`・`test_root.py` 相当の一致，`examples/henon`・`duffing` の分岐図 |
| **0.3.0** | `hes_p`（混合 2 階微分），完全解析 `find_local_bf`，`trace_local_bf`，`trace_local_bf_arclength`（cusp 通過，共鳴フラグ，`:second_multiplier_crossing`），`MappyDSMakieExt`（`plot2d`, `plot_trajectory`） | 0.3.0 | 論文 Fig. 8 / 10 の再現例，Alpazur の PD 曲線，Python 版 193 件に対応するテスト |
| 0.4.x | Julia 固有の拡張: `Symbolics` 拡張，`BifurcationKit` 拡張，`RecipesBase` レシピ，ジェネリック要素型（`BigFloat` 等），マルチスレッド掃引 | — | 需要に応じて |
| **1.0.0** | API 凍結 | — | 0.3 以降に破壊的変更が 1 リリース以上なし，ColPrac 準拠，JuliaHub の docs リンク設定済み，Zenodo DOI 発行 |

各 0.x 内のバグ修正は patch。0.x 間の破壊的変更は CHANGELOG と
TagBot のリリースノートに「breaking」と明記する（AutoMerge の要件）。

## 6. リリース手順（General レジストリ）

### 6.1 初回のみの準備

1. **リポジトリ**: `Yuu-Miino/MappyDS.jl` を作成し，PkgTemplates で雛形生成，
   `Project.toml` の `uuid` は `UUIDs.uuid4()`。
2. **Registrator GitHub App** を `MappyDS.jl` にインストールする
   （General への登録は GitHub.com のパッケージでは Registrator 必須）。
3. **Documenter 用デプロイキー**: `doc.mappy` に書き込み可の deploy key を
   **Python 用 `DOC_DEPLOY_KEY` とは別に**追加し，秘密鍵を base64 で
   `MappyDS.jl` の secret `DOCUMENTER_KEY` に置く（§8.1）。
4. **TagBot** のワークフローに `ssh: ${{ secrets.DOCUMENTER_KEY }}` を渡す。
   既定の `GITHUB_TOKEN` で打ったタグは他のワークフロー（docs 配備）を
   起動できないため。
5. **Codecov**: 公開リポジトリでもトークンが必要なので `CODECOV_TOKEN` を
   secret に置く。
6. **Dependabot**: `.github/dependabot.yml` に `github-actions`（`/`）と
   `julia`（`/`, `/docs`, `/test`）を週次で設定（2025-12 から Julia 対応。
   CompatHelper は保守モードで非推奨）。
7. **名前の事前確認**: `RegistryCI.AutoMerge.meets_distance_check("MappyDS", names)`
   をローカルで実行する。
8. `CITATION.bib`（PkgTemplates `Citation`）と，`mappy` 側と同じ論文を引く
   `CITATION.cff` を置く。Zenodo の GitHub 連携を有効にすると TagBot が
   作る GitHub Release ごとに DOI が発行される。

### 6.2 各リリースの手順（CONTRIBUTING.md 相当）

1. `Project.toml` の `version` を上げる（次の版は既存版から patch / minor /
   major のいずれか +1 でなければ AutoMerge が通らない）。
2. `CHANGELOG.md` の `[Unreleased]` を `## [X.Y.Z] - YYYY-MM-DD` に移す
   （Python 版と同じ Keep a Changelog 形式）。
3. `main` にマージし，そのコミットに
   `@JuliaRegistrator register` とコメントする。リリースノートは
   同じコメントに `Release notes:` 以下に書くと TagBot が GitHub Release に
   転記する。
4. General に PR が立つ。AutoMerge の確認項目: 名前規則，`[compat]` の
   上限，ライセンス，`Pkg.add` と `import` が通ること（compat の最小・最大
   Julia で）。**新規パッケージは 3 日間**，以後の版は 10〜15 分の待機後に
   自動マージ。
5. マージ後 TagBot が `vX.Y.Z` タグと GitHub Release を作成し，その push で
   docs ワークフローが走って `doc.mappy/julia/vX.Y/` と `stable` が更新される。
6. 0.1.0 登録後に JuliaHub の DocumentationGeneratorRegistry に
   `hosted` 方式で `https://yuu-miino.github.io/doc.mappy/julia/stable/`
   を登録する（JuliaHub の自動ビルドではなく自サイトを指すため）。

### 6.3 Python リポジトリ側で必要な変更

- `docs/source/conf.py` のナビゲーションバーに「Julia」リンク
  （`../julia/stable/`）を追加し，`README.md` に Julia 版の存在を記す。
- `tests/` に参照値をエクスポートするスクリプト
  （`tests/export_reference.py` → JSON）を追加し，Julia 側の
  `test/fixtures/reference.json` を生成する（§7.2）。
- `doc.mappy` の `index.html` は Python `latest/` へのリダイレクトのまま。
  Sphinx 側の配備ワークフローは `$REF_NAME` ディレクトリしか消さないので
  `julia/` は保たれる。

## 7. テスト・CI

### 7.1 テスト構成

- `Test` + `ReTestItems.jl`（`@testitem`，並列実行，VS Code 連携）。
- 必須チェック: `Aqua.test_all`（曖昧性・stale deps・compat・型海賊等），
  `JET.test_package`，`Documenter.doctest`。
- 数値検証の 3 層:
  1. **参照値**（Python 版から書き出した JSON）との一致（相対 1e-6）。
  2. **有限差分**（Python `conftest.py` の `fd_jacobian` / `fd_hessian` 移植）。
  3. **ForwardDiff を `solve` に通したオラクル**（1 階のみ。到達時刻の
     微分が 2026-02 以降伝播することを利用）。
- リレー振動子の閉形式 Poincaré 写像で Hessian と `hes_p` まで厳密比較。

### 7.2 Python 版との参照値共有

Python 側 `tests/export_reference.py` が各系・各パラメータ点で
`y`, `jac`, `hes`, `jac_p`, `hes_p`, `mode_history`, 分岐点の `params` / `theta`
を JSON に書き出し，Julia 側 `test/fixtures/reference.json` にコピーする。
両リポジトリで同じファイルを持つことで，どちらかを変えたときに差分が
テストで見える。

### 7.3 クロス検証ジョブ（任意，週次）

`PythonCall.jl` + `CondaPkg.jl`（`CondaPkg.toml` の `[pip.deps]` に
`mappy-ds`）で同一入力を両実装に流して比較する。環境変数
`MAPPYDS_CROSSCHECK=1` のときだけ `@testitem` を有効にし，CI では
`schedule`（週次）と `workflow_dispatch` の別ジョブで回す（conda 解決で
1 分以上かかるので PR ごとには回さない）。libstdc++ 衝突は conda の
Python を使うことで回避する。

### 7.4 CI マトリクス（`CI.yml`）

| 軸 | 値 |
| --- | --- |
| Julia | `lts`（1.10），`1`（1.13），`pre`（1.14 alpha，失敗許容） |
| OS | `ubuntu-latest`；`1` のみ `macos-latest`（arm64）と `windows-latest` も |
| アクション | `actions/checkout@v7`, `julia-actions/setup-julia@v3`, `julia-actions/cache@v3`, `julia-actions/julia-buildpkg@v1`, `julia-actions/julia-runtest@v1`, `julia-actions/julia-processcoverage@v1`, `codecov/codecov-action@v7` |
| 追加ジョブ | docs（`julia-actions/julia-docdeploy@v1` または `docs/make.jl` 直接），doctest，フォーマッタ（`fredrikekre/runic-action@v1` か `julia-actions/julia-format@v4`） |

`push` の `tags: ['*']` を CI に含めること（TagBot 経由の docs 配備に必要）。

## 8. ドキュメント

### 8.1 配備先と Documenter 設定

Documenter 1.20 の `deploydocs` で `doc.mappy` の `gh-pages` ブランチの
`julia/` サブディレクトリに配備する。

```julia
deploydocs(;
    repo = "github.com/Yuu-Miino/MappyDS.jl.git",       # パッケージ側（GITHUB_REPOSITORY と一致させる）
    deploy_repo = "github.com/Yuu-Miino/doc.mappy.git",  # 配備先
    branch = "gh-pages",
    dirname = "julia",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
    push_preview = false,
)
```

- 配備先が別リポジトリなので `GITHUB_TOKEN` では押せず，**SSH の
  `DOCUMENTER_KEY`** が必須（Documenter のホスティング手引きどおり）。
  Documenter 1.20 の `deploy_repo` + `dirname` でこの構成が動くことを
  2026-11-08 に確認済み（`gh-pages` に `julia/dev/`，`julia/versions.js`，
  `julia/index.html` が作られ，Python 側のディレクトリは無傷）。
- 生成されるレイアウトは `julia/stable`（最新版へのシンボリックリンク），
  `julia/v0.1/`，`julia/dev/`，`julia/versions.js`，`julia/index.html`
  （リダイレクト）。Documenter は `dirname` 配下しか消さないので，Python
  側の `v0.3.0/`，`develop/`，`latest` と干渉しない。
- Python 側は `vX.Y.Z` 単位，Julia 側は `vX.Y` 単位のディレクトリになる。
  意図的な差として docs の「バージョン」ページで説明する。

### 8.2 内容

- Python 版の構成（Getting started / User guide（モード概念，導関数，
  周期点と継続，プロット，動作例）/ API reference）をそのまま写す。
- `examples/` は Literate.jl で書き，docs の「Working examples」に
  Markdown 化して載せる（Python 版の `plot_directive` に相当）。
- `DocumenterInterLinks` で Python 版 Sphinx の `objects.inv`
  （`https://yuu-miino.github.io/doc.mappy/latest/objects.inv`）を読み，
  対応する Python API に `@extref` でリンクする。
- `DocumenterCitations` で NOLTA 2024 論文を引用。
- トップページに Python 版との機能対応表（§5）と API 対応表（§4.1）。

## 9. 進め方とスケジュール

実装は Claude のセッションで行う。人手が要るのは §11 の作業（リポジトリ
作成，secrets，Registrator へのコメント，レビュー）だけなので，期間は
「セッション数」と「人手の待ち時間」で見積もる。1 セッションは数時間分の
作業（実装・テスト・docs・コミット）を指す。

| Phase | 内容 | Claude 側 | 人手 |
| --- | --- | --- | --- |
| 0 | 計画確定，Julia 雛形（Project.toml，CI，TagBot，Dependabot，docs/make.jl，README，CHANGELOG），`doc.mappy/julia/dev/` への配備疎通 | **完了（2026-11-08）**: 雛形を `MappyDS.jl` の `main` にプッシュし，CI（LTS / 1.13 / pre，macOS / Windows，Runic，Aqua，JET，ExplicitImports，docs，doctest）が全て成功，`doc.mappy/julia/dev/` に配備済み | リポジトリ作成，deploy key と secrets の設定（§11） |
| 1 | 0.1.0 の実装（functions → modes → maps → root/continuation → docs），Python 側の参照値エクスポートと JSON 共有，登録申請 | **実装完了（2026-11-08，1 セッション）**: `MappyDS.jl` `main` の `Release 0.1.0` コミットで CI 全ジョブ成功（単体テスト 810 件，Python 版の参照値と 1e-6〜1e-7 で一致）。`tests/export_reference.py` を Python 側に追加 | 登録コメント（未），AutoMerge 待機 3 日 |
| 2 | 0.2.0（`jac_p`，擬弧長法） | **実装完了（2026-11-09）**: `MappyDS.jl` `main` の `Release 0.2.0` コミット。補正子は ODE 写像のノイズ床に対応するため「残差が改善しなくなった時点で最良点を採用」する判定を追加 | 登録コメント（0.1.0 の 3 日待機後） |
| 3 | 0.3.0（`hes_p`，分岐曲線，Makie 拡張，例題 5 本），Python 側の docs リンクと README 更新（Python 0.3.1） | **実装完了（2026-11-09）**: `MappyDS.jl` `main` の `Release 0.3.0` コミット（単体テスト 1,860 件，全ジョブ成功）。`examples/` 5 本は Literate.jl で書き，docs の「Working examples」でビルド時に実行（Alpazur の PD 点 B1 ≈ 9.08 を含む）。Python 側は docs のナビと README から Julia 版へリンク済み | 登録コメント（0.1.0 → 0.2.0 → 0.3.0 の順，各 AutoMerge 後） |
| 4 | Discourse の Package announcements，JuliaHub docs 登録，Zenodo，フィードバック対応 | 随時 | 告知文の確認 |
| 5 | 1.0.0（API 凍結の判断） | 1 セッション | 判断 |

各セッションの終わりには CI が緑で，docs の `dev` が更新された状態に
しておく（中断してもどこからでも再開できるようにする）。

Phase 1 の最初のセッションで，`ContinuousCallback` 経由の到達時刻と Python
版の Izhikevich 参照値（y = −3.4783849356）が 1e-6 で一致することを確認する
（最大のリスク §10-1 の早期検証）。本セッションの環境からは Julia を取得
できなかったので，雛形の動作確認は次回セッション（Julia が使える環境）で
行う。

## 10. リスクと対策

1. **数値の不一致**（ソルバ差: `Tsit5` vs scipy `RK45`，イベント時刻の
   精度差）。対策: 参照値は Python 側を `rtol=1e-10` 程度で再生成して
   「真値に近い値」にし，許容差 1e-6 で比較する。閉形式を持つリレー振動子で
   厳密比較する。
2. **AutoMerge の名前チェック**（視覚距離は未計算）。対策: 登録前に
   `meets_distance_check` をローカル実行，代替名 `HybridPoincare` を用意。
3. **読み込み時間と初回実行時間**（OrdinaryDiffEq v7 + ForwardDiff）。
   対策: サブライブラリ限定の依存，`PrecompileTools` の `@compile_workload`
   に Izhikevich 例を入れる，`@time_imports` を CI で記録する。
4. **入れ子 ForwardDiff の落とし穴**（chunk サイズ，タグ衝突，ユーザ関数が
   `Float64` に型固定している場合の `MethodError`）。対策: `DerivativeError`
   に「`y` の要素型を固定しないでください」と案内を入れる，docs に
   ユーザ関数の書き方の章を置く。
5. **SciML の API 変動**（v7 の `VectorContinuousCallback` 符号反転のような
   変更）。対策: `ContinuousCallback` のみ使う，compat 上限で守る，
   Dependabot の PR で CI が検知する。
6. **2 実装の乖離**。対策: 参照値 JSON の共有（§7.2），CHANGELOG を両方で
   同じ形式にする，機能対応表を docs に置く。
7. **secrets の運用ミス**。`DOCUMENTER_KEY` は `doc.mappy` の deploy key
   なので TagBot が `MappyDS.jl` にタグを push するには使えない（0.1.0 の
   初回 TagBot で判明）。TagBot には `MappyDS.jl` 自身の deploy key
   `TAGBOT_KEY` を別に発行する。手順書は `CONTRIBUTING.md` に置いた。
8. **ライセンス**: ModelingToolkit v11 の AGPL 部分に依存しない（§4.2）。

## 11. 人手が必要な作業（チェックリスト）

Claude のセッションからは行えない，または行うべきでない作業。

- [x] GitHub に空リポジトリ `Yuu-Miino/MappyDS.jl` を作成する（MIT，既定
      ブランチ `main`）。2026-11-08 作成，雛形プッシュ済み。
- [ ] Registrator GitHub App（<https://github.com/apps/juliateam-registrator>）
      を `MappyDS.jl` にインストールする。
- [x] `doc.mappy` 用の deploy key を新規発行する（2026-11-08 設定済み。
      docs ジョブから `julia/dev/` への配備を確認）:
      `ssh-keygen -t ed25519 -N "" -C "MappyDS docs deploy" -f mappyds_docs_key`。
      公開鍵を `doc.mappy` の **Settings → Deploy keys** に *Allow write
      access* で追加し，秘密鍵を `base64 -w0 mappyds_docs_key` した値を
      `MappyDS.jl` の secret **`DOCUMENTER_KEY`** に置く。両ファイルは削除。
- [x] Codecov で `MappyDS.jl` を有効にし，トークンを secret
      **`CODECOV_TOKEN`** に置く（2026-11-08 設定済み。CI からの
      アップロード成功を確認）。
- [ ] リポジトリ設定 **Actions → General** で "Allow GitHub Actions to
      create and approve pull requests" を有効にする（Dependabot / TagBot 用）。
- [x] `main` の `Release 0.1.0` コミット（`ebca9d1`）に
      `@JuliaRegistrator register` とコメントする（2026-11-09 実施。登録 PR:
      <https://github.com/JuliaRegistries/General/pull/175403>。新規パッケージは
      AutoMerge まで 3 日待機）。
- [ ] TagBot 用に `MappyDS.jl` 自身の deploy key を発行する（2026-11-12 に
      必要と判明。`ssh-keygen -t ed25519 -N "" -C "MappyDS TagBot" -f mappyds_tagbot_key`，
      公開鍵を **MappyDS.jl** の Settings → Deploy keys に *Allow write access*
      で追加，秘密鍵を secret **`TAGBOT_KEY`** に置く）。その後 Actions の
      TagBot を *Run workflow* で手動実行すると `v0.1.0` が打たれる。
- [ ] 0.1.0 のマージ後，`Release 0.2.0`（`69c009c`），続いて `Release 0.3.0`
      （`9147f6a`）のコミットにも同じコメントをする（既存パッケージの新版は
      通常数十分で AutoMerge）。各版の TagBot がタグとリリースを作り，docs の
      `v0.x.y/` と `stable/` が配備される。
- [ ] 0.1.0 登録後，Zenodo の GitHub 連携を `MappyDS.jl` で有効にする。
- [ ] 0.1.0 登録後，JuliaDocs/DocumentationGeneratorRegistry への PR
      （Claude が用意する）を確認する。

## 12. Phase 4 の準備資料

### 12.1 JuliaHub の docs リンク（DocumentationGeneratorRegistry）

JuliaHub は既定で docs を自前ビルドしようとするが，MappyDS の docs は
`doc.mappy` に配備しているので「hosted」として登録する。
<https://github.com/JuliaDocs/DocumentationGeneratorRegistry> をフォークし，
`Registry.toml` の末尾に次を追記して PR を出す（0.1.0 が General に入った後）:

```toml
[dc692248-7fca-4550-9de6-3b8cf1554f02]
name = "MappyDS"
location = "https://yuu-miino.github.io/doc.mappy/julia/stable"
method = "hosted"
```

PR のタイトルは `Add MappyDS` で足りる。CI が TOML と URL を検査する。

### 12.2 Discourse 告知文案（Package announcements）

タイトル: `[ANN] MappyDS.jl: Poincaré maps of hybrid dynamical systems with analytic derivatives`

本文（英語，そのまま貼れる）:

> MappyDS.jl is a Julia port of the Python package `mappy-ds`. It implements the *mode concept*
> (Miino, NOLTA IEICE 15(2), 2024): a hybrid/piecewise-smooth system is written as a graph of
> *modes* (ODE flows that end on a border, or discrete maps), and the package composes them
> into a Poincaré map `T` together with its derivatives.
>
> What you get for a `PoincareMap` or a plain `Diffeomorphism`:
>
> - `image_detail(T, y0, p)`: the image, the Jacobian `DT` and the Hessian, plus parameter
>   derivatives and mixed second derivatives on request. Flows use the variational equations
>   with the analytic border correction (no AD through the ODE solver); user functions are
>   differentiated with ForwardDiff on `SVector`s.
> - `find_cycle`, `trace_cycle`, `trace_cycle_arclength`: periodic points and their
>   continuation (natural parameter or pseudo-arclength, with fold / period-doubling /
>   Neimark–Sacker detection).
> - `find_local_bf`, `trace_local_bf`, `trace_local_bf_arclength`: local bifurcation points
>   solved from `(Tⁿ(y) − y, det(DTⁿ − e^{iθ}I))` with a fully analytic Jacobian, and their
>   continuation in two parameters (cusps, strong resonances flagged).
> - A Makie extension: `plot_trajectory` and an interactive `plot2d` viewer (GLMakie).
>
> Built on OrdinaryDiffEqTsit5, SciMLBase callbacks, SimpleNonlinearSolve, ForwardDiff and
> StaticArrays. Julia 1.10+.
>
> Docs with five worked examples (Hénon, coupled logistic maps, Izhikevich neuron, forced
> Duffing and Alpazur oscillators): https://yuu-miino.github.io/doc.mappy/julia/stable/
> Repository: https://github.com/Yuu-Miino/MappyDS.jl
> Python package: https://github.com/Yuu-Miino/mappy
>
> Feedback on the API is welcome: the package is at 0.3 and the API will be frozen for 1.0
> once it has settled.

投稿は 0.3.0 が General に入り，`stable/` の docs が 0.3.0 を指してから。

### 12.3 Zenodo

1. <https://zenodo.org/account/settings/github/> で `Yuu-Miino/MappyDS.jl` を ON にする。
2. 以降，TagBot が GitHub Release を作るたびに Zenodo が DOI を発行する
   （既に打たれた `v0.1.0` 等は，Release を一度編集・保存すれば取り込まれる）。
3. 発行された concept DOI を `README.md` のバッジと `CITATION.bib` に入れる
   （Claude が次のセッションで行う）。

### 12.4 登録 PR の確認方法

General の PR には AutoMerge がガイドライン判定をコメントする。「new package」は
判定が全て通っていれば 3 日後に自動マージされる。名前の類似度などで止まった
場合はコメントに理由が出るので，その文面を Claude に渡せば対処案を出す。

## 13. 変更履歴

- 2026-11-12: 0.1.0 が General にマージ（#175403）。直後の TagBot は
  `DOCUMENTER_KEY`（doc.mappy の鍵）では `MappyDS.jl` に push できず失敗。
  TagBot.yml を `TAGBOT_KEY` に切り替え，§10-7 と §11 を訂正。
- 2026-11-09 (3): 0.1.0 の登録 PR（General #175403）が立ち，AutoMerge の
  3 日タイマーが開始。Phase 4 の資料（§12: JuliaHub の docs 登録エントリ，
  Discourse 告知文案，Zenodo 手順）を追加。MappyDS.jl に `CONTRIBUTING.md`
  （開発手順とリリース手順，§6.2 の内容）と Codecov バッジを追加。
- 2026-11-09 (2): Phase 3（0.3.0）の実装完了。`release-0.3` で CI を緑にして
  `main` に ff マージ。直した点: Makie 拡張は ExplicitImports のため
  `using Makie: Axis, ...` と明示し，描画クロージャ名が Makie の `update!` と
  衝突しないよう改名；Makie も `image` をエクスポートするので例題では
  `using MappyDS: image` が必要（plotting ガイドに注記）；Izhikevich の
  `find_local_bf` テストは ODE 許容誤差を 1e-8 / 1e-10 に締める。
  例題は Literate.jl 形式で `examples/` に置き，docs ビルドで実行して
  図と出力を載せる（docs 環境に CairoMakie と StaticArrays を追加）。
  Python 側: `working_examples.rst` から Julia 版の例題ページへリンク。
- 2026-11-09: Phase 2（0.2.0）の実装完了。Phase 3 の数値部分（`hes_p`，`find_local_bf`，
  `trace_local_bf(_arclength)`）を `release-0.3` ブランチで作成中。
- 2026-11-08 (2): Phase 1 の実装完了を記録。CI を緑にするまでに直した点:
  JET は SciML 内部の抽象解析で誤検出が多いため `target_modules` で自パッケージに
  限定，`StepFunctionResult` は非パラメトリック型に，軌道の最終サンプルは
  モード終了状態で上書き，`find_cycle` は特異ヤコビアン等の例外を失敗結果に変換。
- 2026-11-08: 初版。ヒアリングの結果（名前・docs 配備先・`evolve`・
  `param_keys`・登録時期・Julia compat・Runic・実装体制）を反映。
  Phase 0 の雛形を生成して `Yuu-Miino/MappyDS.jl` の `main` にプッシュ。
  CI を緑にするまでに直した点: `Test` stdlib の UUID，JET の Julia 版
  ごとの compat（0.9 が 1.10 用）と `target_defined_modules` の廃止，
  DocumenterInterLinks のプロジェクト名は英数字のみ，ExplicitImports
  対策で依存は `import X` で読み込む，Runic の整形。docs は
  `doc.mappy/julia/dev/` に配備済み。
