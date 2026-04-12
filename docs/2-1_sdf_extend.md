# SDF 拡張メモ

新しい SDF 手法を追加するときの最小手順です。

1. 入力は `LabelVolume` または正規化済み mask にする。
2. 出力 array の軸順と単位を明記する。
3. 小さい synthetic volume で test を追加する。
4. core 関数が安定してから `transform` の feature 出力につなぐ。
5. optional dependency が必要な場合は、分かりやすいエラーにする。

SDF の variant は、まず `transform` 系の feature として追加します。
今後の実装順は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) を優先してください。

scoring に関する処理は SDF 生成側ではなく `wafergeo.compare.metric_*` に置きます。
