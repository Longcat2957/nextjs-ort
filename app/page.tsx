/* eslint-disable */
'use client';

import * as ort from "onnxruntime-web";
import React, { useEffect, useState, useRef } from "react";
import { Progress } from "@nextui-org/react";
import { Spinner } from "@nextui-org/spinner";
import { Card, CardBody, CardHeader } from "@nextui-org/react";
import { Divider } from "@nextui-org/react";

const MAX_WIDTH: number = 224;
const MAX_HEIGHT: number = 224;
const IMAGE_CLASSIFICATION_DOWNLOAD_URL = "https://api.tensorcube.net/server/status/s3/classification";

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

export default function Page() {
  const [isWasmAvailable, setIsWasmAvailable] = useState<boolean | null>(null);
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
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

          <CardBody>
            <p></p>
          </CardBody>
        </Card>
      )}
    </div>
  );
}