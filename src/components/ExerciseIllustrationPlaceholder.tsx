import React from 'react';
import { Activity } from 'lucide-react';

interface ExerciseIllustrationProps {
  exerciseId: string;
  exerciseLabel: string;
  className?: string;
  aspectRatio?: 'video' | 'wide' | 'square';
}

/**
 * Clean, neutral line-vector illustration placeholder for rehabilitation kinematics.
 * Renders an accessible, medical-grade geometric SVG diagram of joint motion paths
 * and allows users to drop in custom SVG / PNG assets easily.
 */
export function ExerciseIllustrationPlaceholder({
  exerciseId,
  exerciseLabel,
  className = '',
  aspectRatio = 'video',
}: ExerciseIllustrationProps) {
  // Geometric line paths representing joint axis and movement vectors
  const renderVectorSchematic = () => {
    switch (exerciseId) {
      case 'squat':
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            {/* Ground line */}
            <line x1="30" y1="130" x2="170" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            {/* Torso & Head */}
            <circle cx="90" cy="30" r="10" stroke="#2563EB" strokeWidth="2.5" />
            <line x1="90" y1="40" x2="80" y2="75" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            {/* Arms forward balance */}
            <line x1="90" y1="50" x2="130" y2="55" stroke="#64748B" strokeWidth="2" strokeLinecap="round" />
            {/* Hip to Knee to Ankle */}
            <line x1="80" y1="75" x2="115" y2="95" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            <line x1="115" y1="95" x2="90" y2="130" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            {/* Joint angle arc marker */}
            <path d="M 105,90 A 15 15 0 0 1 100,105" stroke="#0284C7" strokeWidth="1.5" strokeDasharray="2 2" />
            <text x="125" y="95" fill="#0284C7" fontSize="9" fontFamily="monospace" stroke="none">90°</text>
          </svg>
        );

      case 'shoulder_raises':
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            <line x1="50" y1="130" x2="150" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="100" cy="28" r="9" stroke="#2563EB" strokeWidth="2.5" />
            <line x1="100" y1="37" x2="100" y2="85" stroke="#2563EB" strokeWidth="3" />
            {/* Left arm neutral, Right arm abducted 85° */}
            <line x1="100" y1="45" x2="70" y2="75" stroke="#64748B" strokeWidth="2" />
            <line x1="100" y1="45" x2="155" y2="48" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            {/* Kinetic motion arc */}
            <path d="M 140,80 A 45 45 0 0 0 155,48" stroke="#0284C7" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1="100" y1="85" x2="88" y2="130" stroke="#2563EB" strokeWidth="3" />
            <line x1="100" y1="85" x2="112" y2="130" stroke="#2563EB" strokeWidth="3" />
            <text x="135" y="70" fill="#0284C7" fontSize="9" fontFamily="monospace" stroke="none">85°</text>
          </svg>
        );

      case 'crossover_arm_stretch':
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            <line x1="60" y1="130" x2="140" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="100" cy="25" r="9" stroke="#2563EB" strokeWidth="2.5" />
            <line x1="100" y1="34" x2="100" y2="80" stroke="#2563EB" strokeWidth="3" />
            {/* Stretched arm across chest */}
            <line x1="110" y1="42" x2="60" y2="52" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            {/* Supporting arm */}
            <path d="M 85,65 L 75,45 L 78,35" stroke="#0284C7" strokeWidth="2" strokeLinecap="round" />
            <line x1="100" y1="80" x2="90" y2="130" stroke="#2563EB" strokeWidth="3" />
            <line x1="100" y1="80" x2="110" y2="130" stroke="#2563EB" strokeWidth="3" />
            <text x="45" y="32" fill="#0284C7" fontSize="9" fontFamily="monospace" stroke="none">Transverse Adduction</text>
          </svg>
        );

      case 'calf_raises':
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            <line x1="40" y1="130" x2="160" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="100" cy="22" r="9" stroke="#2563EB" strokeWidth="2.5" />
            <line x1="100" y1="31" x2="100" y2="75" stroke="#2563EB" strokeWidth="3" />
            <line x1="100" y1="75" x2="100" y2="115" stroke="#2563EB" strokeWidth="3" />
            {/* Elevated foot on metatarsals */}
            <line x1="100" y1="115" x2="115" y2="130" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            <line x1="90" y1="130" x2="115" y2="130" stroke="#94A3B8" strokeWidth="1" strokeDasharray="2 2" />
            {/* Elevation vector arrow */}
            <path d="M 85,125 L 85,110 M 82,114 L 85,110 L 88,114" stroke="#0284C7" strokeWidth="1.5" />
            <text x="55" y="115" fill="#0284C7" fontSize="9" fontFamily="monospace" stroke="none">+Elevation</text>
          </svg>
        );

      case 'tree_pose':
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            <line x1="50" y1="130" x2="150" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="100" cy="20" r="8" stroke="#2563EB" strokeWidth="2.5" />
            {/* Prayer hands above or chest */}
            <path d="M 85,38 L 100,28 L 115,38" stroke="#64748B" strokeWidth="2" />
            <line x1="100" y1="28" x2="100" y2="70" stroke="#2563EB" strokeWidth="3" />
            {/* Standing leg */}
            <line x1="100" y1="70" x2="100" y2="130" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            {/* Bent leg resting on inner thigh */}
            <polyline points="100,70 128,88 102,96" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            <text x="135" y="90" fill="#0284C7" fontSize="9" fontFamily="monospace" stroke="none">Balance</text>
          </svg>
        );

      default:
        return (
          <svg viewBox="0 0 200 140" className="w-full h-full max-h-36 mx-auto stroke-blue-600 fill-none">
            <line x1="30" y1="130" x2="170" y2="130" stroke="#CBD5E1" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="95" cy="26" r="9" stroke="#2563EB" strokeWidth="2.5" />
            <line x1="95" y1="35" x2="90" y2="75" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            <line x1="90" y1="75" x2="115" y2="100" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            <line x1="115" y1="100" x2="105" y2="130" stroke="#2563EB" strokeWidth="3" strokeLinecap="round" />
            <line x1="90" y1="75" x2="70" y2="130" stroke="#64748B" strokeWidth="2.5" strokeLinecap="round" />
            <path d="M 120,60 L 140,60 M 135,55 L 140,60 L 135,65" stroke="#0284C7" strokeWidth="1.5" />
            <text x="110" y="50" fill="#0284C7" fontSize="8" fontFamily="monospace" stroke="none">Kinematic Path</text>
          </svg>
        );
    }
  };

  const aspectClass =
    aspectRatio === 'square' ? 'aspect-square' : aspectRatio === 'wide' ? 'aspect-[21/9]' : 'aspect-[16/9]';

  return (
    <div
      className={`relative w-full ${aspectClass} rounded-xl bg-[#F0F7FF]/60 border border-blue-100 flex flex-col items-center justify-center p-3 text-center overflow-hidden group ${className}`}
    >
      {/* Background subtle anatomical grid */}
      <div
        className="absolute inset-0 opacity-[0.25] pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(to right, #93C5FD 1px, transparent 1px), linear-gradient(to bottom, #93C5FD 1px, transparent 1px)',
          backgroundSize: '20px 20px',
        }}
      />

      {/* Vector Line Graphic */}
      <div className="relative z-10 w-full flex items-center justify-center my-auto">
        {renderVectorSchematic()}
      </div>

      {/* Structured Label Badge */}
      <div className="relative z-10 mt-2 flex items-center gap-2 px-2.5 py-1 rounded bg-white/90 border border-blue-200/80 text-[10px] text-slate-600 font-mono tracking-wider uppercase shadow-xs">
        <Activity className="w-3 h-3 text-blue-600 shrink-0" />
        <span>[ LINE VECTOR SCHEMATIC ] • {exerciseLabel}</span>
      </div>
    </div>
  );
}
