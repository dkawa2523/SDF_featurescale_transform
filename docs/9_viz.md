以下は、先ほど提案したグラフ群を **実際に生成してPNG/SVG等へ出力するためのコード実装計画**を、「他の層と同様に Artifact駆動・設定駆動・プラグイン拡張前提」で具体化したものです。
目的は **(a) グラフ作成が再現可能**、**(b) 第三者がグラフの修正・追加を容易にできる**、**(c) 大規模データでも破綻しにくい（要約統計中心＋必要時のみ重いmap出力）** です。

---

# 1. 追加する “Reports/Visualization” サブシステムの位置づけ

* **入力**：各層で既に生成される Artifact（LabelQA/SDFQA/MeshQA/Obs2D/MetricResult/SEMObs/DatasetManifest…）
* **処理**：

  1. Artifactを走査して **統一スキーマのテーブル（DataFrame）** を構築（＝集計・要約）
  2. テーブルを入力として **グラフ（画像）を出力**
  3. 生成結果（画像・テーブル・実行定義・ハッシュ）を **ReportArtifact** として保存
* **出力**：

  * `figures/*.png`（＋任意で`svg`）
  * `tables/*.parquet`（分析しやすい）
  * `report_manifest.json`（再現性・監査）
  * 任意で `index.html`（画像一覧・リンク）

---

# 2. ディレクトリ構成（レビューしやすい責務分割）

`wafergeo/reports/` を新設し、**抽出（extract）**と**描画（plots）**、**実行（runner）**を分離します。

```text
wafergeo/reports/
  README.md                 # 使い方・設計方針・拡張ガイド（必須）
  schema.py                 # ReportSpec/PlotSpec/ReportManifest/PlotResult 等の型
  runner.py                 # ReportRunner（specを読み、extract→plot→export）
  registry.py               # plot/extractor の登録（プラグイン）
  context.py                # ReportContext（ArtifactStore/キャッシュ/ログ/テーマ）
  cache.py                  # table/figure のキャッシュ（hashでskip）
  export/
    image_export.py         # savefig統一（png/svg、dpi、メタ埋め込み）
    table_export.py         # parquet/csv出力
    html_export.py          # index.html生成（任意）
  extract/
    index_build.py          # RunIndex（sample/trial/datasetの対応表）
    label_tables.py         # LabelQA -> DF
    sdf_tables.py           # SDFQA + (必要なら統計) -> DF
    mesh_tables.py          # MeshQA/stats -> DF
    obs_tables.py           # Obs2D QA -> DF
    sem_tables.py           # SEMObs/SEMRaw -> DF
    metrics_tables.py       # MetricResult/EvalResult -> DF
    surrogate_tables.py     # DatasetManifest/splits -> DF
  plots/
    base.py                 # PlotTask I/F、共通ヘルパ（軸/単位/凡例）
    label/
      material_hist.py
      volume_fraction_box.py
      adjacency_heatmap.py
    sdf/
      tsdf_saturation.py
      band_fraction.py
      gradmag_hist.py
      method_bench_scatter.py
    mesh/
      faces_verts_box.py
      degenerate_before_after.py
      interface_area_heatmap.py
    observe/
      mask_area_box.py
      height_stats.py
      loop_count_hist.py
    metrics/
      loss_breakdown.py
      residual_map.py
      chamfer_vs_tsdf_scatter.py
      cd_profile.py
    sem/
      pixel_size_dist.py
      open_contour_rate.py
      transform_dist.py
      overlay_debug.py
    surrogate/
      param_dist.py
      param_corr_heatmap.py
      split_leak_check.py
  tests/
    test_specs/             # 最小specとダミーartifactで回帰テスト
    test_plots_smoke.py     # 代表plotのスモークテスト
```

> **ポイント**
>
> * `extract/*` は「Artifact → 正規化テーブル」だけを担当（副作用少）
> * `plots/*` は「テーブル → figure」だけを担当（基本純粋関数に近い）
> * `runner.py` が orchestration（I/O・キャッシュ）を担当
>   → 第三者は「新しいグラフ＝plotsに1ファイル追加」「新しい集計＝extractに1ファイル追加」で済みます。

---

# 3. 実行ワークフロー（CLI/バッチ）

## 3.1 入力の基準：RunIndex（対応表）を必須にする

「どのsampleがどのArtifact IDを持つか」を、事前に **RunIndex（表）** として与えるのが運用上強いです。
（これが無いと、レポート生成側がストアを総当たり検索して遅くなる・仕様が曖昧になる。）

`RunIndex`（parquet推奨）の例：

| sample_id | group_id | label_artifact_id | sdf_artifact_id | mesh_artifact_id | obs2d_sim_topdown_id | sem_obs_topdown_id | eval_trial_id |
| --------- | -------- | ----------------- | --------------- | ---------------- | -------------------- | ------------------ | ------------- |

* sim-onlyレポートならSEM列は空でOK
* trial（比較）レポートなら `eval_trial_id` を含める

## 3.2 CLI例（設計）

* `wafergeo report build --spec configs/reports/qa_dashboard_v1.yaml --index runs.parquet --out reports/qa_v1/`
* `wafergeo report build --spec configs/reports/assim_debug_v1.yaml --index trials.parquet --out reports/assim_v1/ --return-maps`
* `wafergeo report validate --spec ...`（必要データの存在チェックだけ）

---

# 4. ReportSpec（設定駆動：後から修正しやすい）

## 4.1 YAMLスキーマ（概念）

```yaml
schema_version: "report/v1"
report_id: "qa_dashboard_v1"
title: "QA Dashboard v1"
inputs:
  run_index: "runs.parquet"
filters:
  where:
    - "lot_id == 'L123'"
    - "sdf_method == 'edt_scipy'"
output:
  formats: ["png"]      # ["png","svg"]
  dpi: 200
  write_tables: true
  write_html_index: true
plots:
  - name: "label.material_hist"
    params:
      normalize: "fraction"      # count|fraction
  - name: "label.adjacency_heatmap"
    params:
      aggregate: "sum"           # sum|mean
  - name: "sdf.tsdf_saturation"
    params:
      by_material: true
  - name: "metrics.loss_breakdown"
    params:
      group_by: ["observer","metric"]
      top_k_trials: 50
```

> **設計意図**
>
> * グラフ追加は `plots:` に1行足すだけ
> * 変更もYAMLのparamだけで可能
> * `filters` によりロット/条件/手法別の比較レポートが作りやすい

---

# 5. コアI/F（第三者が “追加しやすい” ための最小契約）

## 5.1 PlotTask（プラグインI/F）

```python
# wafergeo/reports/plots/base.py
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class PlotResult:
    plot_name: str
    figure_paths: list[str]
    tables_written: list[str]
    status: str              # OK/WARN/FAIL
    messages: list[str]
    meta: dict

class PlotTask(Protocol):
    name: str
    version: str
    required_tables: list[str]   # 例: ["label.material_counts", "index.run_index"]

    def run(self, ctx: "ReportContext", params: dict) -> PlotResult:
        ...
```

* 各plotは **必要なテーブル名**を宣言（依存関係が明確）
* `run()` は `ctx.tables["label.material_counts"]` のように参照して描画するだけ

## 5.2 Extractor（テーブル生成のプラグインI/F）

```python
class TableExtractor(Protocol):
    table_name: str
    version: str
    required_inputs: list[str]   # 例: ["index.run_index"]

    def build(self, ctx: "ReportContext") -> "pd.DataFrame":
        ...
```

* ReportRunnerは「plotが必要とするtableが無ければ extractorを呼んで作る」
* テーブルをparquetにキャッシュ → 次回は読み込みで高速化

## 5.3 ReportContext（共通資源）

```python
@dataclass
class ReportContext:
    store: ArtifactStore
    out_dir: str
    spec: ReportSpec
    tables: dict[str, pd.DataFrame]
    cache: ReportCache
    theme: PlotTheme
    logger: Logger
```

---

# 6. 実装の中心：Runner（extract → plot → export → manifest）

## 6.1 Runnerの責務（固定）

1. RunIndex読み込み（必須）
2. filter適用（任意）
3. 必要テーブルをビルド or キャッシュ読込
4. plotごとに `PlotTask.run()` 実行
5. 出力（png/svg/parquet/html）
6. `report_manifest.json` 作成（再現性）

## 6.2 キャッシュ設計（後で“同じ条件なら再生成しない”）

* `spec_hash`（report specのhash）
* `index_hash`（RunIndexのhash）
* `code_version`（git commitやパッケージversion）
* `table_version` / `plot_version`

これらを使って、

* テーブルが同一条件なら再計算しない
* 図が同一条件なら再描画しない
  を実現します（運用で効きます）。

---

# 7. “テーブル生成（extract）” の具体化：最初に作る標準テーブル群

グラフ追加を簡単にするには、まず **横断的に使い回せる正規化テーブル**を作るのが最重要です。
（plot側が毎回Artifactを直接読むと、追加が難しくなります。）

以下を「第一段階の標準テーブル」として固定します。

## 7.1 index.run_index（入力）

* そのまま読み込み（sample/trialの対応）

## 7.2 label.material_counts（#1,#2用）

列例：

* sample_id, material_id, material_name, voxel_count, volume_nm3, fraction

## 7.3 label.adjacency_long（#5用）

列例：

* sample_id, mat_i, mat_j, adjacency_count, adjacency_fraction

## 7.4 sdf.qa_summary（#7,#8,#9用）

列例：

* sample_id, material_id, saturation_ratio, band_fraction, grad_mean, grad_p95, nan_rate

## 7.5 mesh.qa_summary（#13,#14用）

列例：

* sample_id, extractor, num_faces, num_verts, degenerate_ratio_before, degenerate_ratio_after

## 7.6 mesh.interface_area_long（#15用）

列例：

* sample_id, mat_i, mat_j, interface_area_nm2

## 7.7 obs.qa_summary（#18,#20,#21用）

列例：

* sample_id, observer_name, mask_area_nm2, perimeter_nm, num_loops, open_loop_count, tsdf_nan_rate

## 7.8 sem.qa_summary（#22,#23,#24用）

列例：

* sample_id, observer_name, pixel_size_nm, open_rate, tx_nm, ty_nm, rot_deg, scale

## 7.9 metrics.long（#26,#28,#29用）

列例：

* trial_id, sample_id, observer_name, metric_name, loss, value_nm, status

> **これらテーブルを先に固める**と、以降のグラフ追加はほぼ「DataFrame→描画」だけになります。

---

# 8. “グラフ実装（plots）” の具体化：共通作法とテンプレ

## 8.1 共通作法（第三者が迷わないためのルール）

* plotは **入力テーブルのみ**に依存（できるだけ）
* plotは **ファイル名規約**を守る

  * `figures/{plot_name}__{hash_or_tag}.png`
* plotは **メタ情報**を返す（件数、フィルタ条件、軸単位など）
* 失敗しても例外で落とさず、`PlotResult.status="FAIL"` でrunnerに返す（レポート生成全体を止めない）

## 8.2 Plotテンプレ（新規追加時）

```python
# wafergeo/reports/plots/label/material_hist.py
import matplotlib.pyplot as plt
from wafergeo.reports.registry import register_plot

@register_plot("label.material_hist", version="1.0")
class LabelMaterialHist:
    name = "label.material_hist"
    version = "1.0"
    required_tables = ["label.material_counts"]

    def run(self, ctx, params):
        df = ctx.tables["label.material_counts"]

        normalize = params.get("normalize", "fraction")  # "count"|"fraction"
        # 集計（例：dataset全体でmaterial別にsum）
        g = df.groupby(["material_name"], as_index=False).agg({"fraction":"mean","voxel_count":"sum"})

        fig, ax = plt.subplots()
        if normalize == "fraction":
            ax.bar(g["material_name"], g["fraction"])
            ax.set_ylabel("fraction")
        else:
            ax.bar(g["material_name"], g["voxel_count"])
            ax.set_ylabel("voxel_count")

        ax.set_title("Material distribution")
        fig.tight_layout()

        path = ctx.exporter.save_figure(fig, ctx.out_dir, self.name, ctx.spec)
        plt.close(fig)

        return PlotResult(plot_name=self.name, figure_paths=[path], tables_written=[],
                          status="OK", messages=[], meta={"normalize": normalize, "n_samples": df["sample_id"].nunique()})
```

> 重要：`register_plot()` により、runnerは **名前だけでplotを呼べる**（拡張容易）

---

# 9. 画像出力（export/image_export.py）の統一仕様

画像出力は “プロジェクト全体の見た目・再現性” に影響するので、1箇所に集約します。

## 9.1 savefigの標準仕様

* PNG（既定）

  * `dpi = spec.output.dpi`
* SVG（任意）

  * 論文/スライド用途に便利
* 画像メタ（可能なら）

  * spec_hash、plot_name、timestamp をmanifestに保存

## 9.2 出力ディレクトリ規約

```
reports/<report_id>/
  figures/
  tables/
  html/
  report_manifest.json
```

---

# 10. “重い図”（残差マップ・オーバレイ等）の実装方針

残差マップ（Metricsの `maps`）やSEMオーバレイ（画像＋輪郭）は重くなりがちなので、設計として次を守ります。

## 10.1 Heavy plotは “対象サンプルを絞る” を必須化

* `top_k_trials`
* `only_failures`
* `sample_ids`
  など、Specで対象を指定可能にする。

## 10.2 Downsample/clip/ROI を標準化

* residual_map は必要なら `stride` で間引き
* colorbar範囲を `[-p95, p95]` などで自動設定
* 画像overlayは縮小サムネイルも出す（運用で効く）

---

# 11. ReportArtifact（再現性・監査性）

レポートは「いつ・どの定義で」作ったかが重要です。
そのため、`report_manifest.json` を必ず生成します。

## 11.1 manifestに入れるべき内容

* report_id / title / created_at
* spec（全文 or hash + path）
* index_hash / index_path
* 使用した抽出テーブル一覧（table_name、version、行数）
* 作成した図一覧（plot_name、version、path、status）
* エラー/WARN一覧
* git_commit / package_version

---

# 12. テスト方針（第三者が安心して追加できる状態を作る）

## 12.1 スモークテスト（最重要）

* ダミーの小さなRunIndex＋最小Artifact（or mocked ArtifactStore）で

  * `report build` が落ちない
  * 主要plotがpngを吐く

## 12.2 回帰テスト（軽量）

* テーブルextractorの出力カラムが変わったら検知できるよう、
  `expected_columns` をテストに持つ

## 12.3 “追加のしやすさ” のためのテストテンプレ

* `tests/test_specs/minimal_qa.yaml` を用意し、第三者は新plotをそこへ追記してPRで確認できる

---

# 13. 第三者が「グラフ修正/追加」する手順（運用手順として固定）

## 13.1 既存グラフのパラメータ変更

1. `configs/reports/*.yaml` の `plots[].params` を変更
2. `report build` で出力確認
   → コード変更不要

## 13.2 新しいグラフを追加（既存テーブルを使う）

1. `wafergeo/reports/plots/<category>/<name>.py` を追加
2. `register_plot("category.name")` で登録
3. YAMLに `- name: "category.name"` を追記
4. スモークテストに追加（任意だが推奨）

## 13.3 新しい集計テーブルが必要なグラフを追加

1. `wafergeo/reports/extract/<xxx>_tables.py` に extractor を追加
2. `register_extractor(table_name)`
3. plot側で `required_tables` に追加
4. YAMLにplotを追加
5. extractorのカラム定義を README に追記（運用で重要）

---

# 14. 実装の優先順（「まず回る」→「深掘り」）

実装を段階化すると、早期に価値が出て、追加もしやすくなります。

## Phase A（最小で運用が回る：QAダッシュボード）

* extract：`label.material_counts`, `label.adjacency_long`, `sdf.qa_summary`, `mesh.qa_summary`, `obs.qa_summary`, `sem.qa_summary`
* plots：材料ヒスト、界面ヒート、TSDF飽和率、面数分布、mask面積分布、pixel_size分布
* export：png + parquet + manifest + html index（任意）

## Phase B（同化・比較のデバッグ）

* extract：`metrics.long`、（任意）trial timings
* plots：loss内訳、chamfer vs tsdf、CDプロファイル、残差マップ（top_kのみ）

## Phase C（手法比較ベンチ）

* extract：`sdf_bench.long`（method/time/error）
* plots：誤差vs時間散布、方法別箱ひげ

## Phase D（Surrogateデータ監査）

* extract：`surrogate.params_long`, `surrogate.splits`, `surrogate.group_overlap`
* plots：param分布、相関、splitリーク検査

---

## ここまでの要点（この実装計画の芯）

* **テーブル（extract）を正規化してから描画**する設計にする
  → グラフ追加が「DataFrame→matplotlib」で完結し、第三者が迷わない
* **Spec（YAML）で選択・パラメータ調整**できる
  → 修正の多くが設定だけで済む
* **キャッシュ＋manifest**で再現性と運用効率を担保する
* **Heavy plotは対象を絞る**（top_k / failures）を設計として強制できる

---
