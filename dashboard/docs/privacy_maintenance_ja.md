# ダッシュボード公開情報と保守queue

## 公開情報の境界

ダッシュボードの表示APIは、次の情報を返さない。

- host名、IP address、内部port mapping
- `/workspace`を含むlocal filesystem path
- 自動保管の内部保存先path
- worker ID、DB接続文字列、認証情報
- 計算機の配置を特定できるsystem snapshot

実験APIのhardware欄も除外し、成果物APIはtitle、kind、状態、説明、画像previewだけを返す。
preview画像はPostgreSQL `artifact_previews.content bytea`へ保存し、ID指定の画像APIから配信する。

## トップページtimeline

`project_timeline` tableが2026-06-21の1 frame検証開始から現在までを管理する。
各eventは日時、phase、達成内容、定量metric、状態を持つ。トップページは次の累積値も表示する。

- original CPU s20: 10,856.388秒
- NF24 CPU s20: 2,747.270秒
- 累積高速化: 3.95倍
- wall time削減: 74.7%

## NF24

NF24はweightを非線形24-bitで保持する低リソース形式である。各weightはint16 codeと
int8 residual、output方向scaleから復号する。206個のLinear層はGEMM中にweightを復号する。
高メモリprofileは公式FP32 weightへ戻す処理ではなく、同じNF24表現値のQKV/out 56層だけを
起動時にFP32へ展開して常駐させる。

## PostgreSQL maintenance queue

`boat`のevaluation queueを参考に、次を実装した。

- `maintenance_schedules`: 周期、次回時刻、enabled状態
- `maintenance_jobs`: dedupe key、priority、状態、attempt、lease、結果
- `maintenance_job_runs`: attempt単位の実行履歴
- `archived_items`: checksum確認済み退避単位
- `FOR UPDATE SKIP LOCKED`によるatomic claim
- 20秒間隔heartbeatと30分stale lease回収
- 最大3 attempt、失敗時15分backoff

登録済みtaskは未使用成果物の自動保管である。1時間ごとにqueueへ登録し、24時間以上
更新されていない許可済みartifact rootの直下単位を最大4件処理する。保管領域へcopyし、
`rclone check --one-way`が成功した場合だけsourceを削除する。model、checkpoint、backend、inputは
許可rootに含めない。

初回jobはaudit JSON 2件、3,143 byteを退避し、checksum `verified`で完了した。
