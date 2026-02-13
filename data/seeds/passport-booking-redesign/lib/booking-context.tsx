"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

export type ServiceType = "fresh" | "reissue" | "tatkaal-fresh" | "tatkaal-reissue" | null
export type ApplicationType = "normal" | "minor" | "senior" | null

export interface ApplicantDetails {
  fullName: string
  dateOfBirth: string
  gender: string
  email: string
  phone: string
  address: string
  city: string
  state: string
  pincode: string
  idType: string
  idNumber: string
}

export interface SlotDetails {
  pskId: string
  pskName: string
  pskAddress: string
  date: string
  time: string
  slotId: string
}

export interface BookingData {
  serviceType: ServiceType
  applicationType: ApplicationType
  applicant: ApplicantDetails
  slot: SlotDetails | null
  paymentComplete: boolean
  bookingReference: string
}

interface BookingContextType {
  currentStep: number
  setCurrentStep: (step: number) => void
  bookingData: BookingData
  updateServiceType: (type: ServiceType) => void
  updateApplicationType: (type: ApplicationType) => void
  updateApplicantDetails: (details: Partial<ApplicantDetails>) => void
  updateSlotDetails: (slot: SlotDetails) => void
  completePayment: (reference: string) => void
  resetBooking: () => void
  canProceed: (step: number) => boolean
}

const initialApplicant: ApplicantDetails = {
  fullName: "",
  dateOfBirth: "",
  gender: "",
  email: "",
  phone: "",
  address: "",
  city: "",
  state: "",
  pincode: "",
  idType: "",
  idNumber: "",
}

const initialBookingData: BookingData = {
  serviceType: null,
  applicationType: null,
  applicant: initialApplicant,
  slot: null,
  paymentComplete: false,
  bookingReference: "",
}

const BookingContext = createContext<BookingContextType | undefined>(undefined)

export function BookingProvider({ children }: { children: ReactNode }) {
  const [currentStep, setCurrentStep] = useState(1)
  const [bookingData, setBookingData] = useState<BookingData>(initialBookingData)

  const updateServiceType = (type: ServiceType) => {
    setBookingData((prev) => ({ ...prev, serviceType: type }))
  }

  const updateApplicationType = (type: ApplicationType) => {
    setBookingData((prev) => ({ ...prev, applicationType: type }))
  }

  const updateApplicantDetails = (details: Partial<ApplicantDetails>) => {
    setBookingData((prev) => ({
      ...prev,
      applicant: { ...prev.applicant, ...details },
    }))
  }

  const updateSlotDetails = (slot: SlotDetails) => {
    setBookingData((prev) => ({ ...prev, slot }))
  }

  const completePayment = (reference: string) => {
    setBookingData((prev) => ({
      ...prev,
      paymentComplete: true,
      bookingReference: reference,
    }))
  }

  const resetBooking = () => {
    setCurrentStep(1)
    setBookingData(initialBookingData)
  }

  const canProceed = (step: number): boolean => {
    switch (step) {
      case 1:
        return bookingData.serviceType !== null
      case 2:
        return (
          bookingData.applicant.fullName !== "" &&
          bookingData.applicant.dateOfBirth !== "" &&
          bookingData.applicant.email !== "" &&
          bookingData.applicant.phone !== ""
        )
      case 3:
        return bookingData.slot !== null
      case 4:
        return bookingData.paymentComplete
      default:
        return false
    }
  }

  return (
    <BookingContext.Provider
      value={{
        currentStep,
        setCurrentStep,
        bookingData,
        updateServiceType,
        updateApplicationType,
        updateApplicantDetails,
        updateSlotDetails,
        completePayment,
        resetBooking,
        canProceed,
      }}
    >
      {children}
    </BookingContext.Provider>
  )
}

export function useBooking() {
  const context = useContext(BookingContext)
  if (context === undefined) {
    throw new Error("useBooking must be used within a BookingProvider")
  }
  return context
}
