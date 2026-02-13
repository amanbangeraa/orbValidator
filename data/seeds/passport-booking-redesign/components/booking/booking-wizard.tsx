"use client"

import { useBooking } from "@/lib/booking-context"
import { ProgressIndicator } from "@/components/booking/progress-indicator"
import { ServiceSelection } from "@/components/booking/steps/service-selection"
import { ApplicantDetails } from "@/components/booking/steps/applicant-details"
import { SlotSelection } from "@/components/booking/steps/slot-selection"
import { Payment } from "@/components/booking/steps/payment"
import { Confirmation } from "@/components/booking/steps/confirmation"

export function BookingWizard() {
  const { currentStep } = useBooking()

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <ServiceSelection />
      case 2:
        return <ApplicantDetails />
      case 3:
        return <SlotSelection />
      case 4:
        return <Payment />
      case 5:
        return <Confirmation />
      default:
        return <ServiceSelection />
    }
  }

  return (
    <div className="space-y-8">
      {/* Progress indicator - hide on confirmation */}
      {currentStep < 5 && (
        <div className="bg-card rounded-xl border p-6">
          <ProgressIndicator currentStep={currentStep} />
        </div>
      )}

      {/* Step content */}
      <div className="bg-card rounded-xl border p-6 lg:p-8">
        {renderStep()}
      </div>
    </div>
  )
}
