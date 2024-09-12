/* eslint-disable */
'use client';

import * as ort from "onnxruntime-web";
import React, { useEffect, useState, useRef, useCallback } from "react";

// nextui
import { Button } from "@nextui-org/react";
import { Progress } from "@nextui-org/react";
import { Spinner } from "@nextui-org/spinner";
import { Card, CardBody, CardHeader } from "@nextui-org/react";
import { Divider } from "@nextui-org/react";

// icons
import { Upload } from "lucide-react";

// functions
const MAX_WIDTH: number = 224;
const MAX_HEIGHT: number = 224;
const IMAGE_CLASSIFICATION_DOWNLOAD_URL = "https://api.tensorcube.net/server/s3/classification";

function checkWasmAvailability() {
  try {
    if (typeof WebAssembly === 'object' &&
        typeof WebAssembly.instantiate === 'function') {
      const module = new WebAssembly.Module(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00));
      if (module instanceof WebAssembly.Module) {
        return new WebAssembly.Instance(module) instanceof WebAssembly.Instance;
      }
    }
  } catch (e) {
    console.error("Error checking WASM availability:", e);
  }
  return false;
}

async function downloadONNXModel(
  presignedUrl: string,
  onProgress: (progress: number) => void
): Promise<ArrayBuffer> {
  try {
    const pre_response = await fetch(presignedUrl);
    if (!pre_response.ok) {
      throw new Error(`Failed receive presigned_url: ${pre_response.statusText}`);
    }
    
    const response_json = await pre_response.json();
    const actual_url = response_json["presigned_url"];
    
    const response = await fetch(actual_url);
    if (!response.ok) {
      throw new Error(`Failed to download ONNX model: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    const contentLength = +(response.headers.get('Content-Length') ?? '0');
    let receivedLength = 0;
    const chunks: Uint8Array[] = [];

    const updateInterval = 100; // 프로그레스 업데이트 간격 (밀리초)
    let lastUpdateTime = Date.now();

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      const currentTime = Date.now();
      if (currentTime - lastUpdateTime >= updateInterval) {
        const progress = (receivedLength / contentLength) * 100;
        onProgress(Math.min(progress, 99)); // 99%를 최대로 설정
        lastUpdateTime = currentTime;
      }
    }

    const allChunks = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      allChunks.set(chunk, position);
      position += chunk.length;
    }

    onProgress(100); // 다운로드 완료 시 100%로 설정
    return allChunks.buffer;
  } catch (error) {
    console.error('Error downloading ONNX model:', error);
    throw error;
  }
}
async function loadONNXModel(modelWeight: ArrayBuffer): Promise<ort.InferenceSession> {
  try {
    const session = await ort.InferenceSession.create(modelWeight, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    return session;
  } catch (error) {
    console.error("Failed to load ONNX model:", error);
    throw error;
  }
}

// Components

interface ImageUploadProps {
  onImageUploaded: (tensor: ort.Tensor) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUploaded }) => {
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processImage = useCallback(async (file: File) => {
    if (file.size > 5 * 1024 * 1024) {
      setError("파일 크기는 5MB를 초과할 수 없습니다.");
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const img = new Image();
      img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = MAX_WIDTH;
        canvas.height = MAX_HEIGHT;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(img, 0, 0, MAX_WIDTH, MAX_HEIGHT);
        const imageData = ctx?.getImageData(0, 0, MAX_WIDTH, MAX_HEIGHT);

        if (imageData) {
          const rgbData = new Float32Array(3 * MAX_WIDTH * MAX_HEIGHT);
          for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
            rgbData[j] = imageData.data[i] / 255.0;   // R
            rgbData[j + 1] = imageData.data[i + 1] / 255.0; // G
            rgbData[j + 2] = imageData.data[i + 2] / 255.0; // B
          }
          const tensor = new ort.Tensor('float32', rgbData, [1, 3, MAX_HEIGHT, MAX_WIDTH]);
          onImageUploaded(tensor);
        }

        setImage(canvas.toDataURL());
        setError(null);
      };
      img.src = e.target?.result as string;
    };
    reader.onerror = () => {
      setError("이미지 로드 중 오류가 발생했습니다.");
    };
    reader.readAsDataURL(file);
  }, [onImageUploaded]);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processImage(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processImage(file);
    }
  };

  return (
    <Card
      isPressable
      onPress={() => fileInputRef.current?.click()}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className="w-full max-w-3xl mx-auto"
    >
      <CardBody className="flex flex-col items-center justify-center p-8 min-h-[400px]">
        {image ? (
          <div className="w-full h-full flex items-center justify-center">
            <img
              src={image}
              alt="Uploaded image"
              className="max-w-full max-h-[350px] object-contain"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full">
            <Upload className="w-16 h-16 mb-4 text-gray-400" />
            <p className="text-center text-gray-600 text-lg">
              클릭하거나 이미지를 여기에 드래그하세요
            </p>
          </div>
        )}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageChange}
          accept="image/*"
          className="hidden"
        />
        <Button
          color="primary"
          className="mt-4"
          onPress={() => fileInputRef.current?.click()}
        >
          이미지 선택
        </Button>
        {error && <p className="text-red-500 mt-2">{error}</p>}
      </CardBody>
    </Card>
  );
};

// Page

export default function Page() {
  const [isWasmAvailable, setIsWasmAvailable] = useState<boolean | null>(null);
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [inputTensor, setInputTensor] = useState<ort.Tensor | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const smoothProgressRef = useRef<number>(0);

  useEffect(() => {
    const initializeModel = async () => {
      const wasmResult = checkWasmAvailability();
      setIsWasmAvailable(wasmResult);

      if (wasmResult) {
        try {
          const modelBuffer = await downloadONNXModel(IMAGE_CLASSIFICATION_DOWNLOAD_URL, (progress) => {
            // 부드러운 프로그레스 업데이트
            smoothProgressRef.current = progress;
          });
          const loadedModel = await loadONNXModel(modelBuffer);
          setModel(loadedModel);
        } catch (err) {
          const errorMessage = err instanceof Error ? err.message : String(err);
          setError(errorMessage);
        }
      }
    };

    initializeModel();
  }, []);

  useEffect(() => {
    // 프로그레스바를 부드럽게 업데이트하는 애니메이션
    const animateProgress = () => {
      if (downloadProgress < smoothProgressRef.current) {
        setDownloadProgress(prev => Math.min(prev + 1, smoothProgressRef.current));
      }
      requestAnimationFrame(animateProgress);
    };

    const animationId = requestAnimationFrame(animateProgress);
    return () => cancelAnimationFrame(animationId);
  }, []);

  const handleImageUploaded = useCallback((tensor: ort.Tensor) => {
    setInputTensor(tensor);
  }, []);

  return (
    <div className="flex justify-center items-center h-screen">
      {isWasmAvailable === null ? (
        <Spinner size="lg" color="primary" />
      ) : !isWasmAvailable ? (
        <Card>
          <CardBody>
            <p>WebAssembly is not available in your browser. Please use a modern browser that supports WebAssembly.</p>
          </CardBody>
        </Card>
      ) : error ? (
        <Card>
          <CardBody>
            <p>Error: {error}</p>
          </CardBody>
        </Card>
      ) : !model ? (
        <Card className="w-[300px]">
          <CardBody>
            <p className="mb-2">Downloading and loading model...</p>
            <Progress
              aria-label="Downloading ONNX model..."
              value={downloadProgress}
              className="max-w-md"
            />
          </CardBody>
        </Card>
      ) : (
        <Card className="w-[1200px]">
          <CardHeader className="flex gap-3">
            <div className="flex flex-col">
              <p className="text-xl">Image Classification</p>
              <p className="text-small text-default-500">EfficientViT L1 (ImageNet-1k)</p>
            </div>
          </CardHeader>

          <Divider />

          <CardBody className="flex flex-row">
            <div className="flex-1 pr-4">
              <ImageUpload onImageUploaded={handleImageUploaded} />
            </div>
            <Divider orientation="vertical" className="mx-4" />
            <div className="flex-1 pl-4">
              {/* <p className="text-medium text-default-300">Input Tensor Status: {inputTensor ? 'Ready' : 'Not Ready'}</p> */}
              {inputTensor === null ? (
                <p className="text-medium text-default-300">Input Tensor Status : Not ready</p>
              )
              : (
                <div>
                  <p className="text-medium text-default-900">Input Tensor Status : Ready</p>
                  <p className="text-small text-default-400">⭐ Tensor Shape : {inputTensor.dims[0]}, {inputTensor.dims[1]}, {inputTensor.dims[2]}, {inputTensor.dims[3]}</p>
                </div>
              )
              }
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}