# Faire tourner BitNet-TT avec le simulateur ttsim

Ce guide explique comment builder et exécuter ce dépôt en utilisant
[ttsim](https://github.com/tenstorrent/ttsim), le simulateur fonctionnel
Tenstorrent, au lieu d'un vrai chip Blackhole/Wormhole. Il documente l'état
du portage tel que décrit dans [`PORTING_PLAN.md`](../PORTING_PLAN.md), qui
reste la référence pour le détail technique et les limitations connues.

## 0. Où en est le portage

Le backend `ggml-ttnn` (répertoire
[`3rdparty/llama.cpp/ggml/src/ggml-ttnn/`](../3rdparty/llama.cpp/ggml/src/ggml-ttnn/))
sait aujourd'hui :

- ouvrir un device TT-Metalium réel (silicium ou ttsim, choisi par
  `TT_METAL_SIMULATOR`, jamais testé dans le code du backend) ;
- allouer un buffer DRAM par tenseur et y uploader des poids `I2_S`
  (dé-quantifiés en bf16 à l'upload, "Option A") ;
- offloader **uniquement `GGML_OP_MUL_MAT` quand `src0` est `I2_S` et
  `src1`/`dst` sont `F32`**, via `ttnn::matmul`.

Tout le reste (RMSNorm, RoPE, softmax, KV-cache, embeddings...) reste sur
CPU — c'est le comportement normal du graph-splitter de ggml-backend, pas
une limitation à contourner. N'attends pas un forward pass complet sur
device : ce qui est validé aujourd'hui, c'est l'offload du mul_mat pour une
projection isolée.

## 1. Prérequis

Dans cet environnement, tout est déjà en place (vérifié) :

- `TT_METAL_HOME=~/stage_bitnet/Bitnet-TT/tt-metal` — arbre tt-metal
  prébuild, avec `build_Release/` contenant les configs CMake
  (`tt-metalium`, `tt-nn`).
- `~/stage_bitnet/Bitnet-TT/sim/libttsim_bh.so` +
  `~/stage_bitnet/Bitnet-TT/sim/soc_descriptor.yaml` (copie de
  `blackhole_140_arch.yaml`) — le binaire ttsim Blackhole et son descripteur
  SoC, déjà côte à côte comme ttsim l'exige.
- SFPI installé (`/opt/tenstorrent/sfpi`).
- `tt-umd`/`tt-exalens` avec les fixes TTSim (déjà validés : l'exemple
  `metal_example_add_2_integers_in_riscv` tourne proprement sous ttsim dans
  cet environnement).

Si tu repars d'un environnement neuf, suis
[`tt_metal/tt-llk/tests/TTSIM.md`](../../tt-metal/tt_metal/tt-llk/tests/TTSIM.md)
dans le dépôt `tt-metal` pour remettre ça en place.

## 2. Variables d'environnement

```bash
export TT_METAL_HOME=~/stage_bitnet/Bitnet-TT/tt-metal
# Le runtime tt-metal (rtoptions.cpp) ne lit PAS TT_METAL_HOME lui-même :
# sans ça, tout binaire qui n'est pas lancé depuis la racine du dépôt
# tt-metal échoue avec "TT_FATAL: Root Directory is not set."
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME

export TT_METAL_SIMULATOR=~/stage_bitnet/Bitnet-TT/sim/libttsim_bh.so
export TT_METAL_SLOW_DISPATCH_MODE=1     # ttsim ne fait que du slow dispatch
export TT_METAL_DISABLE_SFPLOADMACRO=1   # SFPLOADMACRO pas implémenté par ttsim

# Nécessaire pour tout binaire lancé hors de l'arbre de build tt-metal
export LD_LIBRARY_PATH=$TT_METAL_HOME/build_Release/lib:$LD_LIBRARY_PATH
```

Ces quatre premières variables sont exactement celles utilisées pour
valider le stack ttsim end-to-end dans cet environnement ; les deux
suivantes (`RUNTIME_ROOT`, `LD_LIBRARY_PATH`) sont des pièges spécifiques à
l'exécution *hors* de l'arbre tt-metal (donc pertinents ici, puisque le
binaire final est `BitNet/build/bin/llama-cli`).

## 3. Compiler BitNet avec le backend TTNN

Le CMake du dépôt a un flag dédié, `-DGGML_TTNN=ON`, séparé du build CPU
normal. Reconfigure (le `build/` actuel du dépôt a été généré avec
`GGML_TTNN=OFF`) :

```bash
cd ~/stage_bitnet/Bitnet-TT/BitNet
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_TTNN=ON \
  -DBITNET_X86_TL2=OFF \
  -DTT-Metalium_DIR=$TT_METAL_HOME/build_Release/lib/cmake/tt-metalium \
  -Dtt-nn_DIR=$TT_METAL_HOME/build_Release/lib/cmake/tt-nn \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-cli -j"$(nproc)"
```

Les quatre flags `LLAMA_BUILD_*` sont les mêmes que ceux passés par
`setup_env.py` pour le build CPU normal — sans eux, `LLAMA_BUILD_TOOLS`
vaut `OFF` par défaut (le sous-projet `3rdparty/llama.cpp` n'est pas
"standalone" ici), et `add_subdirectory(tools)` n'est jamais exécuté :
le target `llama-cli` n'existe tout simplement pas et `cmake --build`
échoue avec `No rule to make target 'llama-cli'`.

Build vérifié bout en bout dans cet environnement (~1 min avec cache
compilateur froid) : `llama-cli` compile et link sans erreur avec
`GGML_TTNN=ON`.

`3rdparty/llama.cpp/ggml/src/ggml-ttnn/CMakeLists.txt` sait déjà déduire ces
deux chemins depuis `TT_METAL_HOME` (donc exporte les variables de la
section 2 **avant** de lancer `cmake -B`) — mais seulement si
`TT-Metalium_DIR`/`tt-nn_DIR` ne sont pas déjà en cache. Il y a un second
`tt-metalium-config.cmake` cassé (sans `Metalium.cmake` à côté) directement
sous `$TT_METAL_HOME/build_Release/`, et si une configuration précédente
l'a trouvé en premier (via `CMAKE_PREFIX_PATH`), il reste figé en cache et
le fallback interne du script ne le corrige plus. Passer les deux `-D`
explicitement ci-dessus contourne ça dans tous les cas, y compris en
reconfigurant un `build/` déjà pollué par une tentative précédente.

## 4. Vérifier que le backend se charge sous ttsim

Une fois buildé, `llama-cli` doit énumérer un device `TT_METALIUM0` :

```bash
./build/bin/llama-cli --list-devices
```

Tu dois voir apparaître un device nommé `TT_METALIUM0` (description
`TT_Metalium`), à côté du CPU. Si `TT_METAL_SIMULATOR` est bien positionné,
ce device s'ouvre contre ttsim et non contre du silicium — rien à faire de
plus côté BitNet, le choix silicium/ttsim est entièrement décidé par
tt-metal via cette variable.

Sortie réelle observée dans cet environnement, variables de la section 2
exportées :

```
Available devices:
  TT_METALIUM0: Tenstorrent Blackhole (TT-Metalium/TT-NN backend, registration skeleton) (0 MiB, 0 MiB free)
```

(le "0 MiB" et "registration skeleton" viennent du fait que
`ggml_backend_dev_get_memory` n'est pas encore implémenté pour ce
backend — sans conséquence pour l'énumération ou l'offload.)

Si tu obtiens `TT_FATAL: Root Directory is not set.`, c'est
`TT_METAL_RUNTIME_ROOT` qui manque (section 2).

## 5. Lancer une inférence avec offload sur ttsim

⚠️ `run_inference.py` force `-ngl 0` en dur — il ne testera **jamais**
l'offload TTNN tel quel. Pour exercer le backend, invoque `llama-cli`
directement.

```bash
# Modèle I2_S, ex. téléchargé via setup_env.py comme d'habitude
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

./build/bin/llama-cli \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "You are a helpful assistant" \
  -n 32 -t 2 \
  -ngl 99 -dev TT_METALIUM0
```

`-ngl 99` fait passer les tenseurs de poids I2_S des couches sur le buffer
device du backend TTNN ; le graph-splitter de ggml-backend garde
automatiquement tout ce que `supports_op()` refuse (norms, RoPE, softmax,
KV-cache...) sur CPU. Attends-toi à une génération lente : ttsim est un
simulateur fonctionnel, pas un modèle de perf, et chaque `mul_mat` offloadé
fait actuellement un aller-retour host↔device (le buffer type ne stocke pas
encore de `ttnn::Tensor` natif — voir `PORTING_PLAN.md` §11).

## 6. Limitations connues à garder en tête

Détail complet dans `PORTING_PLAN.md` §9-11 ; résumé :

- **Vues de tenseurs non supportées** : `init_tensor` fait un
  `GGML_ASSERT` si `view_src != NULL`. Pas encore rencontré par le
  chargement des poids, mais bloquant dès qu'un op créant des vues
  (KV-cache typiquement) serait offloadé.
- **Un `MeshBuffer` par tenseur, toujours accédé en entier** — deux bugs
  tt-metal (offset host mal calculé, écritures/lectures "interior" qui
  atterrissent à l'offset 0) empêchent le partage d'un buffer entre
  tenseurs à des sous-offsets, comme le font CUDA/Metal/RPC normalement.
- **Le `MeshDevice` n'est jamais détruit** (fuite volontaire) — un bug
  tt-metal fait planter la destruction après qu'un op a rempli le program
  cache. Sans conséquence pour un process qui se termine de toute façon,
  mais à savoir si tu instrumentes des tests qui recréent des devices en
  boucle dans le même process.
- Ces trois bugs sont documentés comme probablement présents aussi sur
  silicium réel (logique host-side C++, pas un artefact de simulation) —
  pas encore remontés en amont.

## 7. Dépannage rapide

| Symptôme | Cause probable |
|---|---|
| `TT_FATAL: Root Directory is not set.` | `TT_METAL_RUNTIME_ROOT` non exporté (section 2) |
| `error while loading shared libraries: libtt_metal.so` | `LD_LIBRARY_PATH` ne pointe pas vers `$TT_METAL_HOME/build_Release/lib` |
| Pas de `TT_METALIUM0` dans `--list-devices` | build fait sans `-DGGML_TTNN=ON`, ou `find_package(TT-Metalium)`/`find_package(tt-nn)` a échoué au configure (relire les logs `cmake -B`) |
| `include could not find requested file: .../build_Release/Metalium.cmake` | `TT-Metalium_DIR` pointe (souvent via un cache pollué) vers le config cassé à la racine de `build_Release/` au lieu de `build_Release/lib/cmake/tt-metalium/`. Repasse les deux `-DTT-Metalium_DIR=...`/`-Dtt-nn_DIR=...` explicitement (section 3) — un `-D` en CLI écrase toujours le cache. |
| `TT-Metalium_DIR-NOTFOUND` ou un chemin commençant par `/build_Release/...` | `TT_METAL_HOME` n'était pas exporté dans le shell qui a lancé `cmake` — exporte-le d'abord (section 2), dans le **même** shell. |
| `UnimplementedFunctionality: <opcode>` venant de ttsim | gap dans l'ISA simulée, pas un bug BitNet — à isoler et remonter sur [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim/issues) |
| `Getting NOC translation status is not supported...` | `tt-umd` trop ancien (fixes TTSim manquants) |
| Crash à la sortie du process après une inférence réussie | Le bug §11 "destruction de MeshDevice" — le process se termine, sans impact |

Pour tout le reste (compilation SFPI, pytest LLK bas niveau, architectures
supportées), voir directement
[`tt_metal/tt-llk/tests/TTSIM.md`](../../tt-metal/tt_metal/tt-llk/tests/TTSIM.md).
