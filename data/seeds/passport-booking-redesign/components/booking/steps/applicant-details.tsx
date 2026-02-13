"use client"

import { useState } from "react"
import { ArrowLeft, ArrowRight, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { HelpTooltip } from "@/components/booking/help-tooltip"
import { useBooking } from "@/lib/booking-context"
import { cn } from "@/lib/utils"

interface FieldError {
  [key: string]: string
}

const indianStates = [
  "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
  "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
  "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
  "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
  "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh"
]

const idTypes = [
  { value: "aadhaar", label: "Aadhaar Card" },
  { value: "pan", label: "PAN Card" },
  { value: "voter", label: "Voter ID" },
  { value: "driving", label: "Driving License" },
]

export function ApplicantDetails() {
  const { bookingData, updateApplicantDetails, setCurrentStep } = useBooking()
  const [errors, setErrors] = useState<FieldError>({})
  const [touched, setTouched] = useState<{ [key: string]: boolean }>({})

  const validateField = (name: string, value: string): string => {
    switch (name) {
      case "fullName":
        if (!value.trim()) return "Full name is required"
        if (value.length < 3) return "Name must be at least 3 characters"
        return ""
      case "dateOfBirth":
        if (!value) return "Date of birth is required"
        const age = new Date().getFullYear() - new Date(value).getFullYear()
        if (age < 0 || age > 120) return "Please enter a valid date of birth"
        return ""
      case "gender":
        if (!value) return "Please select your gender"
        return ""
      case "email":
        if (!value.trim()) return "Email is required"
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return "Please enter a valid email"
        return ""
      case "phone":
        if (!value.trim()) return "Phone number is required"
        if (!/^[6-9]\d{9}$/.test(value.replace(/\D/g, ""))) return "Enter a valid 10-digit mobile number"
        return ""
      case "address":
        if (!value.trim()) return "Address is required"
        return ""
      case "city":
        if (!value.trim()) return "City is required"
        return ""
      case "state":
        if (!value) return "Please select your state"
        return ""
      case "pincode":
        if (!value.trim()) return "PIN code is required"
        if (!/^\d{6}$/.test(value)) return "Enter a valid 6-digit PIN code"
        return ""
      case "idType":
        if (!value) return "Please select an ID type"
        return ""
      case "idNumber":
        if (!value.trim()) return "ID number is required"
        return ""
      default:
        return ""
    }
  }

  const handleChange = (name: string, value: string) => {
    updateApplicantDetails({ [name]: value })
    if (touched[name]) {
      const error = validateField(name, value)
      setErrors((prev) => ({ ...prev, [name]: error }))
    }
  }

  const handleBlur = (name: string) => {
    setTouched((prev) => ({ ...prev, [name]: true }))
    const value = bookingData.applicant[name as keyof typeof bookingData.applicant]
    const error = validateField(name, value)
    setErrors((prev) => ({ ...prev, [name]: error }))
  }

  const validateAll = (): boolean => {
    const newErrors: FieldError = {}
    const fields = ["fullName", "dateOfBirth", "gender", "email", "phone", "address", "city", "state", "pincode", "idType", "idNumber"]
    
    fields.forEach((field) => {
      const value = bookingData.applicant[field as keyof typeof bookingData.applicant]
      const error = validateField(field, value)
      if (error) newErrors[field] = error
    })

    setErrors(newErrors)
    setTouched(fields.reduce((acc, field) => ({ ...acc, [field]: true }), {}))
    return Object.keys(newErrors).length === 0
  }

  const handleContinue = () => {
    if (validateAll()) {
      setCurrentStep(3)
    }
  }

  const handleBack = () => {
    setCurrentStep(1)
  }

  const FormField = ({
    name,
    label,
    type = "text",
    placeholder,
    helpText,
    example,
  }: {
    name: string
    label: string
    type?: string
    placeholder?: string
    helpText?: string
    example?: string
  }) => (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label htmlFor={name} className="text-sm font-medium">
          {label}
          <span className="text-destructive ml-0.5">*</span>
        </Label>
        {helpText && <HelpTooltip content={helpText} />}
      </div>
      <Input
        id={name}
        name={name}
        type={type}
        value={bookingData.applicant[name as keyof typeof bookingData.applicant]}
        onChange={(e) => handleChange(name, e.target.value)}
        onBlur={() => handleBlur(name)}
        placeholder={placeholder}
        className={cn(
          "h-11 text-base",
          errors[name] && touched[name] && "border-destructive focus-visible:ring-destructive"
        )}
        aria-invalid={errors[name] && touched[name] ? "true" : "false"}
        aria-describedby={errors[name] ? `${name}-error` : example ? `${name}-example` : undefined}
      />
      {example && !errors[name] && (
        <p id={`${name}-example`} className="text-xs text-muted-foreground">
          Example: {example}
        </p>
      )}
      {errors[name] && touched[name] && (
        <p id={`${name}-error`} className="flex items-center gap-1 text-sm text-destructive" role="alert">
          <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
          {errors[name]}
        </p>
      )}
    </div>
  )

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-2xl font-semibold text-foreground text-balance">
          Tell us about yourself
        </h2>
        <p className="mt-2 text-muted-foreground">
          Please provide accurate information as it will appear on your passport.
          All fields marked with <span className="text-destructive">*</span> are required.
        </p>
      </div>

      {/* Personal Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Personal Information</CardTitle>
          <CardDescription>
            Enter your name exactly as you want it to appear on your passport
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <FormField
              name="fullName"
              label="Full Name"
              placeholder="Enter your full name"
              helpText="Enter your name as it appears on your birth certificate or existing passport"
              example="Rahul Kumar Singh"
            />
          </div>
          
          <FormField
            name="dateOfBirth"
            label="Date of Birth"
            type="date"
            helpText="Your date of birth as per official records"
          />

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="gender" className="text-sm font-medium">
                Gender
                <span className="text-destructive ml-0.5">*</span>
              </Label>
            </div>
            <Select
              value={bookingData.applicant.gender}
              onValueChange={(value) => handleChange("gender", value)}
            >
              <SelectTrigger
                id="gender"
                className={cn(
                  "h-11 text-base",
                  errors.gender && touched.gender && "border-destructive focus:ring-destructive"
                )}
              >
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
            {errors.gender && touched.gender && (
              <p className="flex items-center gap-1 text-sm text-destructive" role="alert">
                <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
                {errors.gender}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Contact Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Contact Information</CardTitle>
          <CardDescription>
            We will use these details to send appointment updates
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <FormField
            name="email"
            label="Email Address"
            type="email"
            placeholder="your.email@example.com"
            helpText="We will send your appointment confirmation here"
            example="rahul.singh@email.com"
          />
          
          <FormField
            name="phone"
            label="Mobile Number"
            type="tel"
            placeholder="Enter 10-digit number"
            helpText="We will send SMS updates to this number"
            example="9876543210"
          />
        </CardContent>
      </Card>

      {/* Address */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Present Address</CardTitle>
          <CardDescription>
            This will be used for your passport and to find nearby Passport Seva Kendras
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <FormField
              name="address"
              label="Street Address"
              placeholder="House/Flat No., Street, Landmark"
              example="123, MG Road, Near City Mall"
            />
          </div>
          
          <FormField
            name="city"
            label="City / Town"
            placeholder="Enter your city"
          />

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="state" className="text-sm font-medium">
                State
                <span className="text-destructive ml-0.5">*</span>
              </Label>
            </div>
            <Select
              value={bookingData.applicant.state}
              onValueChange={(value) => handleChange("state", value)}
            >
              <SelectTrigger
                id="state"
                className={cn(
                  "h-11 text-base",
                  errors.state && touched.state && "border-destructive focus:ring-destructive"
                )}
              >
                <SelectValue placeholder="Select state" />
              </SelectTrigger>
              <SelectContent>
                {indianStates.map((state) => (
                  <SelectItem key={state} value={state.toLowerCase()}>
                    {state}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.state && touched.state && (
              <p className="flex items-center gap-1 text-sm text-destructive" role="alert">
                <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
                {errors.state}
              </p>
            )}
          </div>

          <FormField
            name="pincode"
            label="PIN Code"
            placeholder="6-digit PIN"
            example="400001"
          />
        </CardContent>
      </Card>

      {/* Identity Document */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Identity Verification</CardTitle>
          <CardDescription>
            Provide a government-issued ID for verification at the Passport Seva Kendra
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="idType" className="text-sm font-medium">
                ID Type
                <span className="text-destructive ml-0.5">*</span>
              </Label>
              <HelpTooltip content="You must bring this original ID to your appointment" />
            </div>
            <Select
              value={bookingData.applicant.idType}
              onValueChange={(value) => handleChange("idType", value)}
            >
              <SelectTrigger
                id="idType"
                className={cn(
                  "h-11 text-base",
                  errors.idType && touched.idType && "border-destructive focus:ring-destructive"
                )}
              >
                <SelectValue placeholder="Select ID type" />
              </SelectTrigger>
              <SelectContent>
                {idTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.idType && touched.idType && (
              <p className="flex items-center gap-1 text-sm text-destructive" role="alert">
                <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
                {errors.idType}
              </p>
            )}
          </div>

          <FormField
            name="idNumber"
            label="ID Number"
            placeholder="Enter your ID number"
            helpText="Enter the number exactly as it appears on your ID"
          />
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex items-center justify-between pt-4">
        <Button
          variant="outline"
          size="lg"
          onClick={handleBack}
          className="gap-2 bg-transparent"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={handleContinue}
          className="gap-2 px-8"
        >
          Continue to Slot Selection
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  )
}
