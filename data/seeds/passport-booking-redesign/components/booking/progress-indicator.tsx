"use client"

import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface ProgressIndicatorProps {
  currentStep: number
  totalSteps?: number
}

const steps = [
  { number: 1, label: "Service", description: "Choose service type" },
  { number: 2, label: "Details", description: "Your information" },
  { number: 3, label: "Slot", description: "Pick date & location" },
  { number: 4, label: "Payment", description: "Complete payment" },
  { number: 5, label: "Confirm", description: "Booking confirmed" },
]

export function ProgressIndicator({ currentStep, totalSteps = 5 }: ProgressIndicatorProps) {
  return (
    <nav aria-label="Booking progress" className="w-full">
      <ol className="flex items-center justify-between">
        {steps.slice(0, totalSteps).map((step, index) => {
          const isCompleted = currentStep > step.number
          const isCurrent = currentStep === step.number
          const isUpcoming = currentStep < step.number

          return (
            <li key={step.number} className="flex-1 relative">
              <div className="flex flex-col items-center">
                {/* Connector line */}
                {index > 0 && (
                  <div
                    className={cn(
                      "absolute left-0 right-1/2 top-5 h-0.5 -translate-y-1/2",
                      isCompleted || isCurrent ? "bg-primary" : "bg-border"
                    )}
                    style={{ left: "-50%" }}
                    aria-hidden="true"
                  />
                )}
                {index < totalSteps - 1 && (
                  <div
                    className={cn(
                      "absolute left-1/2 right-0 top-5 h-0.5 -translate-y-1/2",
                      isCompleted ? "bg-primary" : "bg-border"
                    )}
                    style={{ right: "-50%" }}
                    aria-hidden="true"
                  />
                )}

                {/* Step circle */}
                <div
                  className={cn(
                    "relative z-10 flex h-10 w-10 items-center justify-center rounded-full border-2 text-sm font-semibold transition-all",
                    isCompleted && "border-primary bg-primary text-primary-foreground",
                    isCurrent && "border-primary bg-card text-primary ring-4 ring-primary/20",
                    isUpcoming && "border-border bg-card text-muted-foreground"
                  )}
                  aria-current={isCurrent ? "step" : undefined}
                >
                  {isCompleted ? (
                    <Check className="h-5 w-5" aria-hidden="true" />
                  ) : (
                    <span>{step.number}</span>
                  )}
                </div>

                {/* Step label */}
                <div className="mt-3 text-center">
                  <p
                    className={cn(
                      "text-sm font-medium",
                      isCurrent ? "text-primary" : isCompleted ? "text-foreground" : "text-muted-foreground"
                    )}
                  >
                    {step.label}
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground hidden sm:block">{step.description}</p>
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      {/* Screen reader announcement */}
      <p className="sr-only">
        Step {currentStep} of {totalSteps}: {steps[currentStep - 1]?.label}
      </p>
    </nav>
  )
}
