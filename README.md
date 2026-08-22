# 髪3 — Kami Hair Solver soft collider experiment

Blender 5.2 の Hair Curves を、回転自由度付きの幾何学的非線形 Cosserat ロッド有限要素として計算する Windows x64 DLL と Blender Extension です。

Version 0.7.2 / ABI 8 の実装済み挙動は、英語版の [Current Specification](CURRENT_SPECIFICATION.md) に記録しています。softコライダーの設計、採用判断、実測値は [Soft Collider Experiment](SOFT_COLLIDER_EXPERIMENT.md) にまとめています。

この実装は PBD／XPBD／位置拘束投影を使用しません。慣性、ロッド弾性、バリア接触をCUDA上の増分ポテンシャルとして評価し、ストランド並列Gauss-Newton、BVH、TOI制限付き可変時間ステップ、CCD line searchで実行可能解を求めます。各時間区間後にGPU接触インパルスで閉じる法線相対速度を除去し、Coulomb摩擦を適用します。CPUソルバーへのフォールバックはありません。

## 構成

- `include/kami_hair_solver.h`: 安定した C ABI
- `src/`: 倍精度の非線形有限要素・接触コア
- `blender_extension/kami_hair_solver/`: 完全日本語 Blender Extension
- `tests/`: コア、初期交差、活性バリア接触、Blender 5.2 実機試験

## Windows での構築

Visual Studio 2022 Developer PowerShell、CUDA 12.9、Eigen 3.3以降を使用します。

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=120 `
  -DKAMI_EIGEN3_INCLUDE_DIR=C:/path/to/eigen/include/eigen3 `
  -DKAMI_BLENDER_EXECUTABLE=C:/path/to/blender.exe
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
cmake --build build-cuda --target blender-extension-test
cmake --build build-cuda --target blender-extension
```

CUDA 12.9、Visual Studio 2022、Compute Capability 12.0を使用します。成果物は `build-cuda/kami_hair_solver.dll` と `build-cuda/packages/kami_hair_solver_3-0.7.2-windows-x64.zip` です。

## Blender での使用

Extension を有効にすると、3D View のサイドバーに「髪3」タブが現れます。Extension ID、Sceneプロパティ、operator IDをv2から分離しているため、安定版のkami-solver-2と同時に有効化できます。

1. `Object.type == "CURVES"` の「入力する髪」を指定します。
2. 必要なら評価後に三角形化できる「衝突メッシュ」を指定します。
3. 「髪を準備」でゼロ長区間、縮退面、初期交差、最小ギャップを検査できます。
4. 「髪を計算」で全フレームをGPUへ転送してからベイクします。転送・フレーム・可変時間ステップ・反復・残り時間をパネルに表示し、Escで中止できます。
5. 計算に失敗した場合は詳細表示を確認し、保持範囲内の「再開フレーム」を指定して数フレーム前から再計算できます。`失敗位置 / -1 / -5 / -10` のショートカット、パラメータ変更履歴、変更種別に応じた巻き戻し警告も表示します。

正常完了またはエラー終了時には `127.0.0.1:8765/UDP` へ `PING` を送ります。応答は待たず、通知先が応答しない場合もシミュレーション結果と表示を変えません。

元 Hair Curves は変更しません。結果はワールド座標の別オブジェクト「髪_計算結果」と、倍精度の `.khc` キャッシュへ保存します。フレーム変更時にキャッシュから元点数の座標を読み戻します。内部細分点を含む有限要素メッシュは DLL 内だけに保持します。

「softコライダー（実験）」を有効にすると、通常は従来どおりアニメーション目標へ剛体的に追従します。目標までの相対運動がTOI制限を必要とするとき、またはhard非線形求解が失敗したとき、コライダー頂点を接触とアンカーばねの連成未知数へ切り替え、BODY側を局所的に退避させます。softは直前の実行可能な髪位置から開始し、速度予測位置は慣性エネルギーの目標として扱います。実際に変形したBODYは別メッシュ「softコライダー_計算結果」と単精度の `.soft-collider` サイドカーへ出力します。元の衝突メッシュは変更しません。

0.7.2ではhardのNewton上限到達も未収束としてsoft再試行へ接続します。相対運動sweepの前に髪の予測移動量を制限し、一要素のBVH候補が過大なら時間幅を縮小して再試行します。softの髪・BODY方向は同じtrust radiusで制限し、BODYのline-search CCDには全頂点最大paddingではなく実際のswept BVHを使います。

「最小動力学長」は0で無効です。正の値を指定すると、それより短いストランドを開始形状の毛先接線方向へ同一材料で非表示延長します。仮想部分は質量、回転慣性、Cosserat弾性、重力を持ちますが、BODY接触には参加しません。結果オブジェクトとキャッシュには元の Hair Curves の点だけを出力します。これは短い自由端の実物理ではなく、可視部分を長いロッドの先頭部分として計算する明示的な代理モデルです。

## 現在の範囲

- RTX 5070 Ti／`sm_120`向けCUDA倍精度実装です。XPBDフォールバックはありません。
- 開始から終了までの毛根目標とコライダー頂点アニメーションを計算前にGPUへ常駐させます。
- コライダーの頂点アニメーションを線形サブステップ補間し、摩擦の相対速度へ含めます。ポリゴントポロジー変化はエラーです。
- 動くBodyは保守的前進で求めた安全TOIまで進み、髪・毛根・コライダーを同じ時間区間で解きます。残り時間を可変区間として継続し、非線形求解失敗だけを局所的に半分へ戻します。区間数は明示的な「可変時間ステップ上限」で制限します。
- 完成フレーム境界の変位・回転・速度を直近の指定数だけメモリへ保持し、失敗後に物理状態を保ったまま指定フレームへ巻き戻せます。
- 固定毛根要素は Dirichlet 境界として扱い、頭皮へ埋め込まれる固定部分を接触候補から除外します。
- 接触可能な初期状態を要求します。初期交差を押し戻して続行しません。
- softコライダーは静的な準静的退避モデルです。BODYの慣性、面内・曲げ弾性、体積保存、自己衝突は持たず、アニメーション目標への面積集中アンカーだけを持ちます。

実シーン（6757本、元74327点、内部265108節点、258351要素、Body 225184頂点）は、新しい接触既定値による1〜30フレームを完走しました。計算は6分36秒、最終GPUフレームは11.79秒、常駐CUDAメモリは約0.986 GiBでした。RTX 5070 Ti 16GB、Blender 5.2での実測であり、性能保証ではありません。

## 謝辞と独立実装

設計検討では、ZOZO, Inc. の公開プロジェクト [ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver) を、特に接触バリア剛性とTOI制限付き時間進行の参考にしました。Kami Hair Solverは同プロジェクトのソースコードを取り込まず、髪用ソルバーとして独立に実装しています。この謝辞は、ZOZO, Inc.によるKami Hair Solverの承認、提携、著作、所有またはその他の権利帰属を示すものではなく、両プロジェクトそれぞれのライセンスと著作権を変更しません。

ライセンスは GPL-3.0-or-later です。Eigen は MPL-2.0 のヘッダーライブラリとして使用します。ライセンス関係の補足は [Third-party notices](THIRD_PARTY_NOTICES.md) を参照してください。
