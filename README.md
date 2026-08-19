# 髪 — Kami Hair Solver

Blender 5.2 の Hair Curves を、回転自由度付きの幾何学的非線形 Cosserat ロッド有限要素として計算する Windows x64 DLL と Blender Extension です。

この実装は PBD／XPBD／位置拘束投影を使用しません。慣性、ロッド弾性、バリア接触をCUDA上の増分ポテンシャルとして評価し、ストランド並列Gauss-Newton、BVH、CCD line searchで実行可能解を求めます。摩擦は非線形系を不安定化させないGPU Coulombインパルスとして各サブステップ後に適用します。CPUソルバーへのフォールバックはありません。

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

CUDA 12.9、Visual Studio 2022、Compute Capability 12.0を使用します。成果物は `build-cuda/kami_hair_solver.dll` と `build-cuda/packages/kami_hair_solver-0.2.0-windows-x64.zip` です。

## Blender での使用

Extension を有効にすると、3D View のサイドバーに「髪」タブが現れます。

1. `Object.type == "CURVES"` の「入力する髪」を指定します。
2. 必要なら評価後に三角形化できる「衝突メッシュ」を指定します。
3. 「髪を準備」でゼロ長区間、縮退面、初期交差、最小ギャップを検査できます。
4. 「髪を計算」で全フレームをGPUへ転送してからベイクします。転送・フレーム・サブステップ・反復・残り時間をパネルに表示し、Escで中止できます。

元 Hair Curves は変更しません。結果はワールド座標の別オブジェクト「髪_計算結果」と、倍精度の `.khc` キャッシュへ保存します。フレーム変更時にキャッシュから元点数の座標を読み戻します。内部細分点を含む有限要素メッシュは DLL 内だけに保持します。

## 現在の範囲

- RTX 5070 Ti／`sm_120`向けCUDA倍精度実装です。XPBDフォールバックはありません。
- 開始から終了までの毛根目標とコライダー頂点アニメーションを計算前にGPUへ常駐させます。
- コライダーの頂点アニメーションを線形サブステップ補間し、摩擦の相対速度へ含めます。ポリゴントポロジー変化はエラーです。
- 動くBodyが髪を横切る場合は、フレームをロールバックしてサブステップ数を最大4倍まで自動的に増やします。
- 固定毛根要素は Dirichlet 境界として扱い、頭皮へ埋め込まれる固定部分を接触候補から除外します。
- 接触可能な初期状態を要求します。初期交差を押し戻して続行しません。

実シーン（6757本、元74327点、内部265108節点、258351要素、Body 225184頂点）の1〜30フレーム実測は、準備18.38秒、計算194.36秒、常駐VRAM 0.963 GiBでした。RTX 5070 Ti 16GB、Blender 5.2、既定設定での値です。

ライセンスは GPL-3.0-or-later です。Eigen は MPL-2.0 のヘッダーライブラリとして使用します。
