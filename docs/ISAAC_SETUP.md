# LLM QP Robotics 설치와 실행

이 가이드는 Ubuntu 22.04 x86_64에서 Isaac Sim 실행기, 로컬 LLM client와 ROS2 Humble을 서로 분리해 설치합니다. 모든 명령은 저장소 루트에서 실행합니다. Isaac 모델·가중치·외부 USD는 설치 시 내려받으며 Python 환경은 복사하지 않고 스크립트로 만듭니다.

```bash
git clone https://github.com/Idea4Future/llm-qp-robotics.git
cd llm-qp-robotics
```

## 설치 조건

- Ubuntu 22.04 x86_64·glibc 2.35 이상, 시스템 Python 3.10
- NVIDIA RTX GPU와 호환 드라이버, Miniconda/Anaconda, `git`·CMake·C++ build 도구
- 시스템 ROS2 Humble과 Nav2 (`/opt/ros/humble`)
- CUDA LLM 사용 시 `nvcc`가 PATH에 있는 CUDA Toolkit. GPU architecture는 해당 GPU에 맞게 지정
- 최초 설치·창고 USD 로드에 인터넷 연결과 충분한 저장 공간

Isaac Sim 5.1의 공식 요구 사항은 최소 RAM 32 GB·VRAM 16 GB를 안내합니다. 작은 VRAM에서의 구동을 보장하지 않으며 장면·카메라·GUI에 따라 요구량이 달라집니다. 드라이버와 GPU 지원은 [공식 요구 사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)을 확인하세요. Linux pip 설치는 Python 3.11·glibc 2.35 이상을 요구합니다. [공식 pip 설치 안내](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)

이 저장소는 Sim 5.1.0/Lab 2.3.0으로 고정합니다. Sim 5.1은 공식 문서상 지원 종료 버전이므로 최신 Sim/Lab을 일부만 섞어 업그레이드하지 않습니다. 시스템 GPU 드라이버·CUDA Toolkit·외부 자산 서버는 프로젝트 lock으로 고정되지 않습니다.

**약관:** Isaac 실행 wrapper인 `run_python.sh`는 `OMNI_KIT_ACCEPT_EULA=YES`를 설정합니다. [NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)를 읽고 동의한 경우에만 실행하세요. 외부 로봇·창고 자산·Qwen 가중치도 각 원본의 사용 조건을 확인합니다. [Isaac EULA 동의 방식](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim)

## 환경 분리

| 실행 환경 | Python·고정 패키지 | 역할 |
| --- | --- | --- |
| `.envs/isaacsim` Conda | Python 3.11, Sim 5.1.0, Lab2.3.0, torch 2.7.0/torchvision 0.22.0 CUDA 12.8 wheel, CVXPY 1.6.5/OSQP 0.6.7.post3 | CPU PhysX·CPU QP·GPU RTX 렌더링 |
| `.venv` | Python 3.10, 표준 라이브러리 client | 명세·LLM HTTP·실행 조율 |
| 시스템 `/usr/bin/python3` | Python 3.10, apt ROS2 Humble | Nav2와 별도 ROS listener·채팅 서버 |
| `third_party/llm_runtime` | 별도 C++ llama.cpp | 기본 CUDA 또는 선택 CPU 추론 |

Sim 터미널에서 `/opt/ros/humble/setup.bash`를 source하지 않습니다. wrapper는 ROS/Python 경로를 분리하며 Sim의 bundled Python 3.11 bridge와 외부 Python 3.10 ROS 노드는 DDS로 연결합니다. Nav2 adapter가 필요한 시스템 Humble 자식만 실행하므로 채팅 실행 전에 별도 Nav2 launch를 띄울 필요가 없습니다.

## 1. 시스템 도구 준비

Conda·NVIDIA 드라이버와 ROS2 Humble은 먼저 설치합니다. Humble apt 저장소가 구성되어 있는 Ubuntu 22.04에서 필요한 도구를 준비합니다.

```bash
sudo apt install python3.10-venv cmake build-essential git \
  ros-humble-navigation2 ros-humble-nav2-bringup

nvidia-smi
/usr/bin/python3 --version
ldd --version
```

이 프로젝트 스크립트는 NVIDIA 드라이버·ROS apt 저장소·CUDA Toolkit을 설치하지 않습니다. CUDA LLM 빌드를 쓰려면 [NVIDIA CUDA 설치 안내](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)에 따라 호환 Toolkit과 host compiler를 준비하고 `nvcc --version`을 확인합니다. PyTorch CUDA 12.8 wheel은 Python용 runtime이며 `nvcc`를 포함하지 않습니다.

## 2. Isaac 환경과 로봇 설치

```bash
bash scripts/isaac/setup.sh
```

위 명령은 다음 네 단계를 순서대로 실행합니다.

```bash
bash scripts/isaac/bootstrap_conda.sh
bash scripts/isaac/install_dependencies.sh
bash scripts/isaac/fetch_robot.sh
bash scripts/isaac/install_command_client.sh
```

Conda를 못 찾으면 `OPTI_CONDA_EXE`를 본인의 실행 파일로 지정합니다.

```bash
OPTI_CONDA_EXE=/path/to/miniconda3/bin/conda \
  bash scripts/isaac/setup.sh
```

기본 Isaac prefix는 `.envs/isaacsim`입니다. 다른 경로를 쓰면 설치와 모든 실행에서 같은 `OPTI_ISAAC_ENV`를 지정합니다. client 설치는 시스템 Python 3.10으로 `.venv`를 만들며 이미 있는 Python 3.10 venv의 패키지는 재설치하지 않습니다.

설치 스크립트는 pip lock·compatibility constraints와 Isaac Lab v2.3.0 revision을 검사하고 마지막에 `pip check`를 실행합니다. CVXPY/OSQP를 임의로 최신 버전으로 올리지 않습니다. [Isaac Lab pip 설치](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)

`fetch_robot.sh`는 공식 RB-Y1 저장소의 revision `2417a2b2c83bc80b3ad605ab14f4d508d90089a9`와 필요한 그리퍼를 `third_party/rby1_isaac`에 가져오고 `robot_assets.json`의 hash를 확인합니다. 원본 자산을 보존하고 파생 장면에서 로컬 제어와 패드 collider를 구성합니다. SDK UDP 통신용 비공개 바이너리나 실제 그리퍼 서버는 필요하지 않습니다.

## 3. 로컬 Qwen과 llama.cpp 준비

공식 Qwen3-4B Q4_K_M GGUF를 다운로드하고 llama.cpp를 빌드합니다.

```bash
.venv/bin/python scripts/local_llm_setup.py model

# RTX 4060 Ti 예시: compute capability 8.9 → architecture 89
.venv/bin/python scripts/local_llm_setup.py runtime \
  --cuda --cuda-architecture 89
```

다른 GPU에는 해당 architecture 값을 지정합니다. CUDA 빌드는 PATH의 `nvcc`를 사용합니다. 모델 revision·SHA256과 runtime tag는 `scripts/local_llm_setup.py`, 실행 설정은 `configs/local_llm.json`에 있습니다. 모델은 `third_party/llm_models`, runtime은 `third_party/llm_runtime`에 생성됩니다. 추가 학습·API key·외부 LLM 서비스 없이 사용합니다.

CUDA Toolkit 없이 LLM만 CPU로 실행하려면 다음처럼 빌드합니다.

```bash
.venv/bin/python scripts/local_llm_setup.py runtime
```

이후 CLI에 `--backend cpu`를 지정합니다. 웹 worker의 기본 backend는 CUDA이며 **CPU LLM 옵션도 Isaac RTX 렌더링의 GPU 요구를 없애지 않습니다.** worker가 필요한 동안만 loopback LLM 서버를 시작하고 종료하므로 별도 서버를 수동 실행하지 않습니다. LLM 서버를 정리한 뒤 Isaac 실행기를 시작합니다.

## 4. 작은 장면부터 확인

```bash
bash scripts/isaac/run_demo.sh --scene minimal --headless
```

GUI로 보려면 `--headless`를 뺍니다. 이 데모는 로봇·손목 카메라·앱 연결용 제자리 동작이며 LLM·집기·운반을 포함하지 않습니다. 새 `runs/isaac_*` 폴더에 생성되는 요약과 이미지를 확인합니다.

외부 창고 USD 연결은 다음 명령으로 확인합니다. 자산 서버 접근과 추가 렌더 메모리가 필요합니다.

```bash
bash scripts/isaac/run_demo.sh --scene warehouse
```

설치 명령 성공, 앱 초기화, 물리 작업 완료는 각각 확인합니다. 첫 시작의 extension 다운로드와 shader 준비에는 시간이 걸릴 수 있습니다.

## 5. 채팅 실행

```bash
/usr/bin/python3 scripts/run_chat.py --port 8765 --open-browser
```

`http://127.0.0.1:8765`에서 접속합니다. 서버는 로컬 주소에만 bind하고 한 번에 한 작업을 허용합니다. 요청 예시는 다음과 같습니다.

- `A 선반의 빨간 용기를 출고대에 가져다 놓아줘.`
- `B 선반의 파란 용기를 출고대에 옮겨줘.`
- `C 선반의 초록 용기를 출고 구역에 천천히 가져다 놓아줘.`

매 명령은 home의 새 초기 장면에서 한 용기를 오른팔로 출고대 첫 번째 자리에 옮깁니다. 직전 세계 상태를 이어받거나 임의 물체·여러 자리·작업 순서를 지원하지 않습니다. 선반·색 불일치와 지원하지 않는 요청은 거부합니다. LLM의 ID·6가중치·6제약을 검토한 뒤 프로그램이 고정 스킬로 확장합니다. 최신 물체 pose는 실행 중 손목 RGB/PnP로 확인합니다.

화면의 진행 메시지·명세·활성 QP 수식·실행 영상은 해당 worker 이벤트에서 나옵니다. 계획 단계에는 적용 전 QP 수식을 실행했다고 표시하지 않습니다. 물리 요약·독립 판정·종료 코드0을 함께 확인해야 완료를 표시합니다. `실행 중지`는 해당 작업의 자식 프로세스 정리까지 기다립니다.

기본은 headless·보행자 없음입니다. `Isaac Sim 창 보기`와 `이동하는 사람 포함`은 실행 전에 고릅니다. 보행자를 켜면 1명 또는 3명을 선택하며 옵션은 실행 중 잠깁니다. 보행자 OFF이면 인원 설정은 실행에 사용하지 않습니다.

## 6. CLI와 직접 물리 실행

자연어 worker는 새 출력·로그 이름으로 실행합니다.

```bash
mkdir -p logs
bash scripts/isaac/run_command.sh --layout rack-v1 \
  --command 'B 선반의 파란 용기를 출고대에 옮겨줘.' \
  --output runs/command_new > logs/command_new.log 2>&1
```

| 옵션 | 동작 |
| --- | --- |
| `--show-sim` | Isaac GUI. 기본 headless |
| `--moving-person --people-count 3` | 보행자3명. 허용 인원1/3,1명은P3 |
| `--backend cpu` | CPU llama.cpp 사용. Isaac에는 GPU 필요 |
| `--plan-only` | 계획만 검토. 최종 물리 `accepted=false`, 종료 코드2 |
| `--layout legacy` | 기존 A입고대→B조립대 template와 registry 선택 |

직접 물리 실행의 기본 layout은 `legacy`이므로 선반에는 `rack-v1`을 명시합니다. 다음 명령은 LLM 없이 기본 QP 계수로 실행합니다.

```bash
bash scripts/isaac/run_python.sh scripts/isaac/run_transport.py \
  --headless --mode full --scene warehouse --layout rack-v1 --rack A --video \
  --output runs/transport_new > logs/transport_new.log 2>&1
```

`--rack B/C`로 다른 선반을 고릅니다. `--mode load --controller qp --vision`은 이미 선택 선반에 도킹한 정지 상태에서 트레이 적재만 실행합니다. `full`은 손목 카메라와 팔·베이스 QP를 활성화하고 home 접근·적재·출고 운반을 연결합니다. `--headless`를 빼면 GUI입니다. `--output`은 새 폴더를 요구합니다.

독립 정지 집기·유지·놓기 명령은 다음과 같습니다. DLS IK 기반이며 LLM·QP·트레이·주행을 포함하지 않습니다.

```bash
bash scripts/isaac/run_grasp.sh --headless
```

## 센서·사람·물리 모델

베이스 localization과 카메라 세계 pose/extrinsic은 GT입니다. 손목 RGB ArUco/PnP로 용기 pose를 구하며 marker17/18/19가 각각 A빨강/B파랑/C초록에 대응합니다. Depth는 품질 gate이고 직접 물체 좌표를 LLM에 제공해 맞추는 방식이 아닙니다.

전·후방 LiDAR는 PhysX 충돌 장면을 raycast하는 이상적인 프로젝트 센서입니다. RTX 잡음·반사·지연 모델이나 원본 RB-Y1 센서라고 설명하지 않습니다. Nav2 NavFn 전역 경로와 프로젝트의 국소 QP를 사용하며 Nav2 DWB controller·AMCL·SLAM을 실행하는 구조가 아닙니다.

보행자는 `robot-priority-yield-v1`의 scripted kinematic 시나리오입니다. 서로 다른 lane을 걷다가 로봇에 양보해 감속·자기 lane 내 후퇴·대기·재개합니다. 로봇 GT 상태·예정 경로·목적지 예약은 사람 시나리오 행동에만 제공하고 사람 GT pose·속도·양보 target은 로봇 QP에 넣지 않습니다. 로봇은 collision-path label이 붙은 LiDAR 표면점과 자신의 측정 상태를 사용합니다.

사람 외관은 코드로 생성한 안전모·조끼 인체이며 외부 사람 USD가 필요하지 않습니다. 접촉은 높이 1.7 m·반경 .22 m 단일 kinematic capsule이고 팔다리는 시각 애니메이션입니다. 질량 속성 70 kg과 별개로 kinematic 접촉은 사실상 무한 질량입니다. 렌더 직전에 native pose와 표시용 각도를 Visual root에 반영하며 물리 target과 분리합니다. 일반 사람 인식·인간 보행 제어·학습한 사회 행동·의도 예측·동적 재계획을 구현하지 않습니다.

동선은 정적 형상·사람 간격·config의 로봇 고정 대기자리를 검사해 선택합니다. 조작 중 collider 외형도 주기적으로 표본 검사하지만 모든 중간 자세를 연속 보증하지 않습니다. active risk의 반사점이 사라지면 대기하며 관측한 모든 risk가 신선한 positive 관측으로 해소되어야 재개합니다.

용기는 40×50×60 mm·100 g 자유 강체, 트레이는 1 kg, 손목 카메라 장착물은 100 g 설계입니다. 후방 고정 skid는 friction=.02·combine=min, 팔 self-collision은 비활성이며 그리퍼 패드 collider를 보완합니다. 실행 중 물체 부착·중력 제거·물체 pose 덮어쓰기는 사용하지 않습니다.

## 저장 파일과 완료 확인

웹 작업은 `runs/chat/<작업ID>/`, CLI는 지정한 `--output`에 저장합니다. 설치 로그는 `logs/isaac_setup/`에 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `command_summary.json`, `task_spec.json`, `events.jsonl` | 명령 상태·검토한 수치 명세·단계 이벤트 |
| `physics/summary.json`, `physics/trace_validation.json` | 물리 요약·저장 상태 독립 판정 |
| `physics/replay.mp4` | 실제 렌더 프레임. 시뮬레이션 시간 기준 영상 |
| `physics/source_observation.json`, `physics/tray_observation.json` | RGB/PnP·Depth 품질 확인 |
| `physics/state_samples.json`, `physics/navigation_trace.json` | 조작·주행 저장 상태 |

독립 재평가는 새 출력 파일로 실행합니다.

```bash
/usr/bin/python3 scripts/isaac/validate_transport.py \
  --input runs/transport_new --mode full --output runs/transport_new/recheck.json
```

주행 도착·보행자 대기·최종 정착은 명령값뿐 아니라 실제 측정 속도와 일정 구간의 위치·자세 변화를 확인합니다. 정지 명령, 정착 확인, 재출발 명령과 실제 움직임은 별도 이벤트입니다. 구체적인 기준은 `scripts/isaac/physical_rest.py`에서 확인할 수 있습니다.

물리 적분 2 ms·QP 명령 20 ms는 설계 간격입니다. 영상 재생 속도나 기록 시각을 실제 계산 속도·하드 실시간 성능으로 해석하지 않습니다. 완료 판정은 지원된 장면·가정의 작업 확인이며 일반 성공률·모든 자세 무충돌·하드웨어 안전성을 보장하지 않습니다.

## ROS clock 연결

별도 Humble listener와 Sim bundled bridge의 DDS 연결을 확인합니다.

```bash
ROS_DOMAIN_ID=172 \
  bash scripts/isaac/run_demo.sh --scene minimal --headless --seconds 5 --check-ros-clock
```

Sim wrapper는 bundled Humble bridge 라이브러리를 설정하고 외부 listener는 시스템 Python 3.10으로 실행합니다. 이는 `/clock` 연결 확인이며 Nav2 전체 topic loop나 센서 기반 localization의 완료가 아닙니다. 별도 ROS simulation-time 노드는 `use_sim_time=true`를 설정합니다. [공식 standalone ROS bridge 안내](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_python.html)

## 오류 해결

| 증상 | 확인할 사항 |
| --- | --- |
| Conda 미발견 | `OPTI_CONDA_EXE` 경로·실행 권한 |
| `nvcc` 미발견 | 별도 CUDA Toolkit과 PATH. CPU LLM은 CUDA 없이 빌드 |
| CUDA architecture 오류 | `--cuda-architecture`가 GPU에 맞는지 확인 |
| `rclpy` ABI/import 오류 | Isaac 3.11·시스템 ROS 3.10을 분리하고 ROS를 source한 Sim 터미널 사용 금지 |
| 자산 로드·앱 초기화 정체 | 인터넷·asset 서버·extension 다운로드·shader 준비·해당 작업 로그 |
| 렌더러·메모리 오류 | GPU 드라이버·VRAM·작은 minimal/headless 장면부터 확인 |
| 출력 폴더 거부 | 기존 출력·로그 대신 새 이름 지정 |
| GUI STOP/reset 이후 오류 | native PhysX view가 무효이면 작업 중단. 새 초기 장면에서 다시 시작 |

소스를 바꾸면 현재 작업을 종료한 뒤 채팅 서버도 다시 시작합니다. 실행 중인 작업을 새 소스로 실행한 것으로 취급하지 않습니다. native 측정이 무효이면 고정 USD pose로 대체하지 않습니다. 명세나 물리 거부를 조건의 임의 완화로 숨기지 않으며 실패 메시지와 로그를 확인합니다.
