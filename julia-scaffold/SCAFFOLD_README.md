# MappyDS.jl の雛形（一時ディレクトリ）

`docs/JULIA_RELEASE_PLAN.md` の Phase 0 で生成した `Yuu-Miino/MappyDS.jl` の
初期ファイル一式。**このディレクトリは `mappy` リポジトリには残さない。**
新リポジトリへ移したら削除する。

## 移し方（次回の Claude セッション，または手動）

```sh
git clone git@github.com:Yuu-Miino/MappyDS.jl.git
cp -R julia-scaffold/. MappyDS.jl/      # SCAFFOLD_README.md は除く
cd MappyDS.jl && rm -f SCAFFOLD_README.md
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl       # ローカルビルド（deploydocs は CI 外では何もしない）
git add -A && git commit -m "Package skeleton" && git push
```

その後 `mappy` 側で `git rm -r julia-scaffold` する。

## この環境で確認できなかったこと（次回セッションで最初に確認）

本セッションからは Julia を取得できなかったため，以下は未実行。

1. `Pkg.instantiate()` / `Pkg.test()` が通ること（依存の UUID は General
   レジストリの `Package.toml` から取得済み。`[compat]` は 2026-11-08 時点の
   最新版: ForwardDiff 1.4.7, StaticArrays 1.9.24, SciMLBase 3.58.0,
   OrdinaryDiffEqTsit5 2.1.5, OrdinaryDiffEqVerner 2.4.2,
   SimpleNonlinearSolve 2.14.6, PrecompileTools 1.3.4, Makie 0.24.17,
   Aqua 0.8.18, JET 0.13.1, Documenter 1.20.0, DocumenterInterLinks 1.1.0,
   DocumenterCitations 1.5.0, Literate 2.21.0, JSON3 1.14.4,
   ReTestItems 1.35.3, ExplicitImports 1.15.0）。
2. `Documenter.deploydocs` の `deploy_repo` キーワードがインストールされた
   版に存在すること（`?deploydocs`）。無い場合は Python 側の
   `deploy_docs.yaml` と同様に，ビルド成果物を `doc.mappy` に手動で
   push するジョブに置き換える。
3. `DocumenterInterLinks` が Sphinx の `objects.inv`
   （`doc.mappy/latest/objects.inv`）を読めること。
4. `runic-action` と `julia-docdeploy` の入力名が現行版と一致すること。
5. パッケージ UUID `dc692248-7fca-4550-9de6-3b8cf1554f02` は本セッションで `uuid4` 生成したもの。
   そのまま使ってよい。

## secrets とリポジトリ設定

計画書 §11 のチェックリストを参照（`DOCUMENTER_KEY`，`CODECOV_TOKEN`，
Registrator App，Actions の PR 作成許可）。
