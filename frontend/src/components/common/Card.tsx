import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}

export default function Card({ children, className = "", onClick }: CardProps) {
  return (
    <div
      onClick={onClick}
      className={`bg-white rounded-2xl border border-surface-200 p-6 shadow-sm transition-all duration-200 ${
        onClick ? "cursor-pointer hover:shadow-md hover:border-primary-200" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}
