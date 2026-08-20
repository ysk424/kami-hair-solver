# 髪 — Kami Hair Solver

Blender 5.2 の Hair Curves を、回転自由度付きの幾何学的非線形 Cosserat ロッド有限要素として計算する Windows x64 DLL と Blender Extension です。

Version 0.6.2 の実装済み挙動は、英語版の [Current Specification](CURRENT_SPECIFICATION.md) に記録しています。これは初期要求ではなく、現行コードのas-built仕様です。

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

CUDA 12.9、Visual Studio 2022、Compute Capability 12.0を使用します。成果物は `build-cuda/kami_hair_solver.dll` と `build-cuda/packages/kami_hair_solver-0.6.2-windows-x64.zip` です。

## Blender での使用

Extension を有効にすると、3D View のサイドバーに「髪」タブが現れます。

1. `Object.type == "CURVES"` の「入力する髪」を指定します。
2. 必要なら評価後に三角形化できる「衝突メッシュ」を指定します。
3. 「髪を準備」でゼロ長区間、縮退面、初期交差、最小ギャップを検査できます。
4. 「髪を計算」で全フレームをGPUへ転送してからベイクします。転送・フレーム・可変時間ステップ・反復・残り時間をパネルに表示し、Escで中止できます。
5. 計算に失敗した場合は詳細表示を確認し、保持範囲内の「再開フレーム」を指定して数フレーム前から再計算できます。`失敗位置 / -1 / -5 / -10` のショートカット、パラメータ変更履歴、変更種別に応じた巻き戻し警告も表示します。

正常完了またはエラー終了時には `127.0.0.1:8765/UDP` へ `PING` を送り、`PONG` 応答を確認します。通知用アプリが応答しない場合は、シミュレーション結果を変えずに通知エラーをパネルへ表示します。

元 Hair Curves は変更しません。結果はワールド座標の別オブジェクト「髪_計算結果」と、倍精度の `.khc` キャッシュへ保存します。フレーム変更時にキャッシュから元点数の座標を読み戻します。内部細分点を含む有限要素メッシュは DLL 内だけに保持します。

「最小動力学長」は0で無効です。正の値を指定すると、それより短いストランドを開始形状の毛先接線方向へ同一材料で非表示延長します。仮想部分は質量、回転慣性、Cosserat弾性、重力を持ちますが、BODY接触には参加しません。結果オブジェクトとキャッシュには元の Hair Curves の点だけを出力します。これは短い自由端の実物理ではなく、可視部分を長いロッドの先頭部分として計算する明示的な代理モデルです。

## 現在の範囲

- RTX 5070 Ti／`sm_120`向けCUDA倍精度実装です。XPBDフォールバックはありません。
- 開始から終了までの毛根目標とコライダー頂点アニメーションを計算前にGPUへ常駐させます。
- コライダーの頂点アニメーションを線形サブステップ補間し、摩擦の相対速度へ含めます。ポリゴントポロジー変化はエラーです。
- 動くBodyは保守的前進で求めた安全TOIまで進み、髪・毛根・コライダーを同じ時間区間で解きます。残り時間を可変区間として継続し、非線形求解失敗だけを局所的に半分へ戻します。区間数は明示的な「可変時間ステップ上限」で制限します。
- 完成フレーム境界の変位・回転・速度を直近の指定数だけメモリへ保持し、失敗後に物理状態を保ったまま指定フレームへ巻き戻せます。
- 固定毛根要素は Dirichlet 境界として扱い、頭皮へ埋め込まれる固定部分を接触候補から除外します。
- 接触可能な初期状態を要求します。初期交差を押し戻して続行しません。

実シーン（6757本、元74327点、内部265108節点、258351要素、Body 225184頂点）の1〜30フレーム実測は、準備18.38秒、計算194.36秒、常駐VRAM 0.963 GiBでした。RTX 5070 Ti 16GB、Blender 5.2、既定設定での値です。

ライセンスは GPL-3.0-or-later です。Eigen は MPL-2.0 のヘッダーライブラリとして使用します。
