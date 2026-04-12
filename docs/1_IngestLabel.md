# 入力 label volume

`transform`, `compare`, `batch-compare` は、simulation 側の入力を
`LabelVolume` という内部形式に変換してから処理します。

## 対応している入力

| kind | 内容 |
|---|---|
| `npz_label` | NumPy npz の label volume |
| `vti_label` | VTI の label volume |

target 側も `npz_label` / `vti_label` を使えます。
外部輪郭データを target にする場合は `contour_json` を使います。

## npz_label の形式

```text
labels: integer array, shape [X,Y,Z]
spacing: optional float array, shape [3], order [X,Y,Z], default [1,1,1]
origin: optional float array, shape [3], order [X,Y,Z], default [0,0,0]
material_ids: optional integer array
material_names: optional string array
void_id: optional integer
```

ユーザー向けの npz は `[X,Y,Z]` 順です。
内部では次の形式に変換します。

```text
material_id: [Z,Y,X]
spacing: [Z,Y,X]
origin: [Z,Y,X]
```

## void_id

void は「真空」「背景」「比較対象外の空領域」を表します。
material id `0` が存在する場合、デフォルトでは `0` を void として扱います。

material id `0` が存在しない場合は、YAML または npz 内で `void_id` を指定してください。
背景を勝手に推測すると評価結果が不安定になるため、明示指定を必須にしています。

## 設計ルール

- 軸変換は loader の中だけで行う。
- SDF、mesh、metric は raw file ではなく `LabelVolume` を受け取る。
- 新しい入力形式を追加するときは、まず `LabelVolume` へ変換する loader を追加する。
