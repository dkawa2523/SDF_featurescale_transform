# SDF 拡張メモ

新しい SDF 手法を追加するときの最小手順です。

1. 入力は `LabelVolume` または正規化済み mask にする。
2. 出力 array の軸順と単位を明記する。
3. 小さい synthetic volume で test を追加する。
4. core 関数が安定してから `transform` の feature 出力につなぐ。
5. optional dependency が必要な場合は、分かりやすいエラーにする。

SDF の variant だけのために新しい public pipeline は追加しません。
まずは `features.use` または metric の追加で対応してください。

scoring に関する処理は SDF 生成側ではなく `wafergeo.compare.metric_*` に置きます。
