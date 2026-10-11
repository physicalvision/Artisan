// Verified against the 2026-10-11 Artisan.pdf: Tables 1–3 and Figure 6. Null means unavailable, never zero.
const benchmarkData = {
  "physical": {
    "label": "Physical plausibility",
    "source": "Table 2",
    "page": 8,
    "default": "single",
    "metrics": [
      {
        "key": "single",
        "label": "Single-joint collision",
        "direction": "down",
        "decimals": 1,
        "family": "Collision",
        "dataset": "ArtVIP \u00b7 collision under motion",
        "unit": "%"
      },
      {
        "key": "double",
        "label": "Double-joint collision",
        "direction": "down",
        "decimals": 1,
        "family": "Collision",
        "dataset": "ArtVIP \u00b7 collision under motion",
        "unit": "%"
      },
      {
        "key": "multiple",
        "label": "Multi-joint collision",
        "direction": "down",
        "decimals": 1,
        "family": "Collision",
        "dataset": "ArtVIP \u00b7 collision under motion",
        "unit": "%"
      },
      {
        "key": "rest",
        "label": "Rest-state stability",
        "direction": "up",
        "decimals": 0,
        "family": "Stability",
        "dataset": "50 stability prompts \u00b7 gravity simulation",
        "unit": "%"
      },
      {
        "key": "open",
        "label": "Open-state stability",
        "direction": "up",
        "decimals": 0,
        "family": "Stability",
        "dataset": "50 stability prompts \u00b7 gravity simulation",
        "unit": "%"
      },
      {
        "key": "color",
        "label": "Base color PSNR",
        "direction": "up",
        "decimals": 2,
        "family": "Material",
        "dataset": "ArtVIP \u00b7 PBR material buffers",
        "unit": " dB"
      },
      {
        "key": "metallic",
        "label": "Metallic PSNR",
        "direction": "up",
        "decimals": 2,
        "family": "Material",
        "dataset": "ArtVIP \u00b7 PBR material buffers",
        "unit": " dB"
      },
      {
        "key": "roughness",
        "label": "Roughness PSNR",
        "direction": "up",
        "decimals": 2,
        "family": "Material",
        "dataset": "ArtVIP \u00b7 PBR material buffers",
        "unit": " dB"
      },
      {
        "key": "mnre",
        "label": "Mass MnRE",
        "direction": "up",
        "decimals": 3,
        "family": "Mass",
        "dataset": "50 ABO objects \u00b7 reference masses",
        "unit": ""
      },
      {
        "key": "ade",
        "label": "Mass ADE",
        "direction": "down",
        "decimals": 2,
        "family": "Mass",
        "dataset": "50 ABO objects \u00b7 reference masses",
        "unit": " kg"
      },
      {
        "key": "ape",
        "label": "Mass APE",
        "direction": "down",
        "decimals": 3,
        "family": "Mass",
        "dataset": "50 ABO objects \u00b7 reference masses",
        "unit": ""
      }
    ],
    "rows": [
      {
        "method": "NeRF2Physics",
        "model": "",
        "values": {
          "single": null,
          "double": null,
          "multiple": null,
          "rest": null,
          "open": null,
          "color": null,
          "metallic": null,
          "roughness": null,
          "mnre": 0.601,
          "ade": 10.78,
          "ape": 0.509
        }
      },
      {
        "method": "PUGS",
        "model": "",
        "values": {
          "single": null,
          "double": null,
          "multiple": null,
          "rest": null,
          "open": null,
          "color": null,
          "metallic": null,
          "roughness": null,
          "mnre": 0.412,
          "ade": 44.5,
          "ape": 3.016
        }
      },
      {
        "method": "Instruct-Particulate",
        "model": "",
        "values": {
          "single": 70.4,
          "double": 90.4,
          "multiple": 98.7,
          "rest": 68.0,
          "open": 42.0,
          "color": 12.0,
          "metallic": 18.02,
          "roughness": 10.4,
          "mnre": null,
          "ade": null,
          "ape": null
        }
      },
      {
        "method": "PhysX-Anything",
        "model": "",
        "values": {
          "single": 58.4,
          "double": 77.6,
          "multiple": 81.3,
          "rest": 74.0,
          "open": 22.0,
          "color": 10.06,
          "metallic": null,
          "roughness": null,
          "mnre": 0.302,
          "ade": 198.07,
          "ape": 19.292
        }
      },
      {
        "method": "URDF-Anything+",
        "model": "",
        "values": {
          "single": 84.6,
          "double": 99.3,
          "multiple": 100.0,
          "rest": 64.0,
          "open": 48.0,
          "color": null,
          "metallic": null,
          "roughness": null,
          "mnre": null,
          "ade": null,
          "ape": null
        }
      },
      {
        "method": "Articulate-Anything",
        "model": "",
        "values": {
          "single": 31.2,
          "double": 39.2,
          "multiple": 32.8,
          "rest": 64.0,
          "open": 44.0,
          "color": 7.92,
          "metallic": null,
          "roughness": null,
          "mnre": null,
          "ade": null,
          "ape": null
        }
      },
      {
        "method": "Articraft",
        "model": "GPT-5.5",
        "values": {
          "single": 44.5,
          "double": 57.5,
          "multiple": 79.4,
          "rest": 82.0,
          "open": 78.0,
          "color": 10.75,
          "metallic": null,
          "roughness": null,
          "mnre": null,
          "ade": null,
          "ape": null
        }
      },
      {
        "method": "Articraft",
        "model": "GPT-6 Astra",
        "values": {
          "single": 45.5,
          "double": 60.7,
          "multiple": 78.8,
          "rest": 82.0,
          "open": 82.0,
          "color": 11.75,
          "metallic": null,
          "roughness": null,
          "mnre": null,
          "ade": null,
          "ape": null
        }
      },
      {
        "method": "Codex",
        "model": "GPT-6 Astra",
        "values": {
          "single": 32.4,
          "double": 47.0,
          "multiple": 63.7,
          "rest": 86.0,
          "open": 82.0,
          "color": 9.98,
          "metallic": null,
          "roughness": null,
          "mnre": 0.763,
          "ade": 6.63,
          "ape": 0.303
        }
      },
      {
        "method": "Claude",
        "model": "Fable 5.1",
        "values": {
          "single": 39.9,
          "double": 51.7,
          "multiple": 64.7,
          "rest": 84.0,
          "open": 86.0,
          "color": 10.5,
          "metallic": null,
          "roughness": null,
          "mnre": 0.743,
          "ade": 6.51,
          "ape": 0.388
        }
      },
      {
        "method": "Agy",
        "model": "Gemini-Flash-3.8",
        "values": {
          "single": 46.3,
          "double": 65.6,
          "multiple": 81.5,
          "rest": 80.0,
          "open": 78.0,
          "color": 10.15,
          "metallic": null,
          "roughness": null,
          "mnre": 0.775,
          "ade": 5.49,
          "ape": 0.296
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.5 \u00b7 without physics",
        "values": {
          "single": 37.5,
          "double": 47.3,
          "multiple": 77.6,
          "rest": 84.0,
          "open": 82.0,
          "color": 10.69,
          "metallic": 16.32,
          "roughness": 12.97,
          "mnre": 0.742,
          "ade": 6.07,
          "ape": 0.34
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.5",
        "values": {
          "single": 3.5,
          "double": 6.4,
          "multiple": 12.3,
          "rest": 96.0,
          "open": 94.0,
          "color": 11.48,
          "metallic": 16.28,
          "roughness": 13.05,
          "mnre": 0.72,
          "ade": 7.22,
          "ape": 0.385
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.6 Sol",
        "values": {
          "single": 1.9,
          "double": 4.8,
          "multiple": 16.4,
          "rest": 98.0,
          "open": 96.0,
          "color": 10.57,
          "metallic": 15.98,
          "roughness": 14.32,
          "mnre": 0.67,
          "ade": 8.25,
          "ape": 0.566
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-6 Astra",
        "values": {
          "single": 1.7,
          "double": 3.7,
          "multiple": 13.0,
          "rest": 98.0,
          "open": 96.0,
          "color": 11.9,
          "metallic": 16.38,
          "roughness": 12.36,
          "mnre": 0.773,
          "ade": 5.86,
          "ape": 0.26
        }
      }
    ],
    "note": "All 15 rows from Table 2, including backbone variants and the physics ablation. An em dash means the paper reports no result; that row is unranked for the selected metric."
  },
  "spatial": {
    "label": "Spatial fidelity",
    "source": "Table 1",
    "page": 7,
    "default": "iou",
    "metrics": [
      {
        "key": "iou",
        "label": "Appearance IoU",
        "direction": "up",
        "decimals": 4,
        "family": "Appearance",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "ssim",
        "label": "SSIM",
        "direction": "up",
        "decimals": 4,
        "family": "Appearance",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "psnr",
        "label": "PSNR",
        "direction": "up",
        "decimals": 2,
        "family": "Appearance",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": " dB"
      },
      {
        "key": "lpips",
        "label": "LPIPS",
        "direction": "down",
        "decimals": 4,
        "family": "Appearance",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "clip",
        "label": "CLIP",
        "direction": "up",
        "decimals": 4,
        "family": "Appearance",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "rest_ocd",
        "label": "Rest oCD",
        "direction": "down",
        "decimals": 4,
        "family": "Rest-pose geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "rest_pcd",
        "label": "Rest pCD",
        "direction": "down",
        "decimals": 4,
        "family": "Rest-pose geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "rest_piou",
        "label": "Rest pIoU",
        "direction": "up",
        "decimals": 4,
        "family": "Rest-pose geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "open_ocd",
        "label": "Open oCD",
        "direction": "down",
        "decimals": 4,
        "family": "Articulated geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "open_pcd",
        "label": "Open pCD",
        "direction": "down",
        "decimals": 4,
        "family": "Articulated geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      },
      {
        "key": "open_piou",
        "label": "Open pIoU",
        "direction": "up",
        "decimals": 4,
        "family": "Articulated geometry",
        "dataset": "ArtVIP \u00b7 50 reference assets",
        "unit": ""
      }
    ],
    "rows": [
      {
        "method": "Instruct-Particulate",
        "model": "",
        "values": {
          "iou": 0.6843,
          "ssim": 0.8581,
          "psnr": 17.12,
          "lpips": 0.2705,
          "clip": 0.9167,
          "rest_ocd": 0.0027,
          "rest_pcd": 0.1143,
          "rest_piou": 0.2436,
          "open_ocd": 0.001,
          "open_pcd": 0.1132,
          "open_piou": 0.2732
        }
      },
      {
        "method": "PhysX-Anything",
        "model": "",
        "values": {
          "iou": 0.5259,
          "ssim": 0.8457,
          "psnr": 15.26,
          "lpips": 0.3899,
          "clip": 0.8474,
          "rest_ocd": 0.0098,
          "rest_pcd": 0.2703,
          "rest_piou": 0.0965,
          "open_ocd": 0.0123,
          "open_pcd": 0.2989,
          "open_piou": 0.092
        }
      },
      {
        "method": "URDF-Anything+",
        "model": "",
        "values": {
          "iou": 0.6294,
          "ssim": 0.8598,
          "psnr": 16.11,
          "lpips": 0.3819,
          "clip": 0.8319,
          "rest_ocd": 0.0062,
          "rest_pcd": 0.2913,
          "rest_piou": 0.1262,
          "open_ocd": 0.0068,
          "open_pcd": 0.3034,
          "open_piou": 0.1236
        }
      },
      {
        "method": "Articulate-Anything",
        "model": "",
        "values": {
          "iou": 0.6054,
          "ssim": 0.8197,
          "psnr": 14.44,
          "lpips": 0.3893,
          "clip": 0.8444,
          "rest_ocd": 0.0087,
          "rest_pcd": 0.1923,
          "rest_piou": 0.1223,
          "open_ocd": 0.0097,
          "open_pcd": 0.2076,
          "open_piou": 0.1254
        }
      },
      {
        "method": "Articraft",
        "model": "GPT-5.5",
        "values": {
          "iou": 0.7195,
          "ssim": 0.858,
          "psnr": 17.48,
          "lpips": 0.2781,
          "clip": 0.9033,
          "rest_ocd": 0.0026,
          "rest_pcd": 0.0211,
          "rest_piou": 0.2795,
          "open_ocd": 0.004,
          "open_pcd": 0.0235,
          "open_piou": 0.2731
        }
      },
      {
        "method": "Articraft",
        "model": "GPT-6 Astra",
        "values": {
          "iou": 0.7602,
          "ssim": 0.8754,
          "psnr": 18.62,
          "lpips": 0.2347,
          "clip": 0.9197,
          "rest_ocd": 0.0022,
          "rest_pcd": 0.0135,
          "rest_piou": 0.3244,
          "open_ocd": 0.0034,
          "open_pcd": 0.0229,
          "open_piou": 0.3224
        }
      },
      {
        "method": "Codex",
        "model": "GPT-6 Astra",
        "values": {
          "iou": 0.7504,
          "ssim": 0.87,
          "psnr": 18.16,
          "lpips": 0.2486,
          "clip": 0.9169,
          "rest_ocd": 0.0032,
          "rest_pcd": 0.0206,
          "rest_piou": 0.3331,
          "open_ocd": 0.0039,
          "open_pcd": 0.0265,
          "open_piou": 0.3095
        }
      },
      {
        "method": "Claude",
        "model": "Fable 5.1",
        "values": {
          "iou": 0.7316,
          "ssim": 0.8767,
          "psnr": 18.21,
          "lpips": 0.2619,
          "clip": 0.9048,
          "rest_ocd": 0.0015,
          "rest_pcd": 0.0283,
          "rest_piou": 0.3309,
          "open_ocd": 0.005,
          "open_pcd": 0.0407,
          "open_piou": 0.3024
        }
      },
      {
        "method": "Agy",
        "model": "Gemini-Flash-3.8",
        "values": {
          "iou": 0.7542,
          "ssim": 0.8771,
          "psnr": 18.01,
          "lpips": 0.2562,
          "clip": 0.913,
          "rest_ocd": 0.001,
          "rest_pcd": 0.0129,
          "rest_piou": 0.3553,
          "open_ocd": 0.005,
          "open_pcd": 0.0235,
          "open_piou": 0.3113
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.5 \u00b7 without physics",
        "values": {
          "iou": 0.7196,
          "ssim": 0.8669,
          "psnr": 17.64,
          "lpips": 0.2784,
          "clip": 0.91,
          "rest_ocd": 0.0021,
          "rest_pcd": 0.0162,
          "rest_piou": 0.2974,
          "open_ocd": 0.0052,
          "open_pcd": 0.026,
          "open_piou": 0.2825
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.5",
        "values": {
          "iou": 0.7374,
          "ssim": 0.8552,
          "psnr": 17.57,
          "lpips": 0.2826,
          "clip": 0.9086,
          "rest_ocd": 0.0016,
          "rest_pcd": 0.0199,
          "rest_piou": 0.2956,
          "open_ocd": 0.0024,
          "open_pcd": 0.0248,
          "open_piou": 0.2937
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-5.6 Sol",
        "values": {
          "iou": 0.7283,
          "ssim": 0.8701,
          "psnr": 17.63,
          "lpips": 0.2721,
          "clip": 0.915,
          "rest_ocd": 0.0017,
          "rest_pcd": 0.0179,
          "rest_piou": 0.2995,
          "open_ocd": 0.0026,
          "open_pcd": 0.0249,
          "open_piou": 0.2991
        }
      },
      {
        "method": "Artisan",
        "model": "GPT-6 Astra",
        "values": {
          "iou": 0.7675,
          "ssim": 0.8769,
          "psnr": 18.65,
          "lpips": 0.2319,
          "clip": 0.9239,
          "rest_ocd": 0.0011,
          "rest_pcd": 0.0101,
          "rest_piou": 0.3587,
          "open_ocd": 0.0021,
          "open_pcd": 0.016,
          "open_piou": 0.3417
        }
      }
    ],
    "note": "All 13 rows and 11 metrics from Table 1 on ArtVIP. Geometry is evaluated in rest and articulated states. Backbone variants and the physics ablation are listed separately."
  },
  "robotics": {
    "label": "Downstream tasks",
    "source": "Table 3(a)",
    "page": 9,
    "default": "twin",
    "metrics": [
      {
        "key": "twin",
        "label": "RoboTwin success",
        "direction": "up",
        "decimals": 0,
        "family": "Policy success",
        "dataset": "3 RoboTwin 2.0 tasks \u00b7 held-out native test assets",
        "unit": "%"
      },
      {
        "key": "casa",
        "label": "RoboCasa success",
        "direction": "up",
        "decimals": 0,
        "family": "Policy success",
        "dataset": "8 RoboCasa365 tasks \u00b7 held-out native test assets",
        "unit": "%"
      }
    ],
    "rows": [
      {
        "method": "Native only",
        "model": "\u03c00.5 training assets",
        "values": {
          "twin": 80,
          "casa": 39
        }
      },
      {
        "method": "Articraft only",
        "model": "\u03c00.5 training assets",
        "values": {
          "twin": 60,
          "casa": 5
        }
      },
      {
        "method": "Artisan only",
        "model": "\u03c00.5 training assets",
        "values": {
          "twin": 63,
          "casa": 27
        }
      },
      {
        "method": "Native + Artisan",
        "model": "\u03c00.5 training assets",
        "values": {
          "twin": 84,
          "casa": 43
        }
      }
    ],
    "note": "These rows compare training-asset compositions for the same \u03c00.5 policy, not standalone generation methods. All policies are tested on held-out native assets."
  },
  "human": {
    "label": "Human preference",
    "source": "Figure 6",
    "page": 10,
    "default": "top1",
    "metrics": [
      {
        "key": "top1",
        "label": "Top-1 preference",
        "direction": "up",
        "decimals": 1,
        "family": "Human preference",
        "dataset": "Interactive human evaluation",
        "unit": "%"
      },
      {
        "key": "top3",
        "label": "Top-3 preference",
        "direction": "up",
        "decimals": 1,
        "family": "Human preference",
        "dataset": "Interactive human evaluation",
        "unit": "%"
      }
    ],
    "rows": [
      {
        "method": "Artisan",
        "model": "",
        "values": {
          "top1": 43.3,
          "top3": 85.1
        }
      },
      {
        "method": "Codex",
        "model": "",
        "values": {
          "top1": 21.1,
          "top3": 68.7
        }
      },
      {
        "method": "Articraft",
        "model": "",
        "values": {
          "top1": 9.8,
          "top3": 52.4
        }
      },
      {
        "method": "Claude",
        "model": "",
        "values": {
          "top1": 14.2,
          "top3": 43.3
        }
      },
      {
        "method": "Agy",
        "model": "",
        "values": {
          "top1": 10.5,
          "top3": 37.5
        }
      },
      {
        "method": "Instruct-Particulate",
        "model": "",
        "values": {
          "top1": 1.1,
          "top3": 13.1
        }
      }
    ],
    "note": "Participants inspect rendered images and manipulate six anonymized candidates in MuJoCo before ranking their top three. Values are selection rates from Figure 6 (Section 5.2)."
  }
};
