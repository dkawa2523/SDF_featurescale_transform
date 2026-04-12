# Mesh

mesh は `transform` で出力できる特徴量の 1 つです。
比較 metric の主導線は 2D view / SDF / contour ですが、mesh はサロゲート学習や
外部解析に渡す形状特徴として使えます。

## 公開出力

`features.use` に `mesh` を含めると、次を出力します。

```text
features/mesh.npz
features/mesh_summary.json
```

例:

```yaml
task: transform

features:
  use: [sdf, mesh, contour, slice]
```

## 現在の backend

通常導線では、決定的に動く `naive_interface` backend を使います。
VTK / PyVista などの重い可視化・検証用途は optional dependency として扱います。

## 拡張ルール

- mesh 抽出は YAML parsing と分離する。
- 新しい mesh 手法は mesh domain 側に backend として追加する。
- `transform` は出力を保存するだけにし、metric 計算を混ぜない。
- mesh 専用 pipeline は、明確なユーザー workflow が出るまで追加しない。
