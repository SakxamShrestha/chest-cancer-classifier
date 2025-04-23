// src/web/frontend/types/api.ts

export interface PredictionResponse {
    status: string;
    filename: string;
    prediction: string;
    confidence: number;
    probabilities: {
      [key: string]: number;
    };
    timestamp: string;
  }
  
  export interface ClassesResponse {
    status: string;
    classes: string[];
  }
  
  export interface ModelInfoResponse {
    status: string;
    model_loaded: boolean;
    model_path: string;
    model_exists: boolean;
    project_root: string;
    device: string;
    message: string;
  }