"use client";

import { useEffect, useRef, useState } from "react";

type Detection = {
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
};

export default function Overlay({
  imageUrl,
  detections,
}: {
  imageUrl: string;
  detections: Detection[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // When image loads, capture its natural size
  const onImgLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
  };

  // Draw boxes whenever detections or image size changes
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    if (!imgSize.w || !imgSize.h) return;

    // Match canvas to the *rendered* image size
    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scale from original image coords -> displayed coords
    const sx = canvas.width / imgSize.w;
    const sy = canvas.height / imgSize.h;

    // Draw each box
    detections.forEach((d) => {
      const [x1, y1, x2, y2] = d.bbox_xyxy;

      const dx1 = x1 * sx;
      const dy1 = y1 * sy;
      const dw = (x2 - x1) * sx;
      const dh = (y2 - y1) * sy;

      // Box
      ctx.lineWidth = 2;
      ctx.strokeStyle = "lime";
      ctx.strokeRect(dx1, dy1, dw, dh);

      // Label background
      const label = `${d.class_name} ${(d.confidence).toFixed(2)}`;
      ctx.font = "12px system-ui";
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(dx1, dy1 - 16, textW + 6, 16);

      // Label text
      ctx.fillStyle = "white";
      ctx.fillText(label, dx1 + 3, dy1 - 4);
    });
  }, [detections, imgSize]);

  return (
    <div className="relative rounded-xl border border-slate-800 bg-slate-950/40 p-3">
      <div className="mb-2 text-xs text-slate-400">Detections overlay</div>

      <div className="relative">
        <img
          ref={imgRef}
          src={imageUrl}
          onLoad={onImgLoad}
          alt="overlay"
          className="w-full max-h-[320px] object-contain rounded-lg border border-slate-800 bg-black"
        />
        <canvas
          ref={canvasRef}
          className="absolute left-0 top-0 h-full w-full pointer-events-none"
        />
      </div>
    </div>
  );
}
