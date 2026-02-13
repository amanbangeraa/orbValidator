"use client"

import { useState } from "react"
import { ArrowLeft, CreditCard, Building2, Calendar, Clock, User, Shield, Loader2, Lock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Separator } from "@/components/ui/separator"
import { useBooking } from "@/lib/booking-context"
import { cn } from "@/lib/utils"

const paymentMethods = [
  { id: "card", label: "Credit / Debit Card", icon: CreditCard },
  { id: "upi", label: "UPI", icon: Shield },
  { id: "netbanking", label: "Net Banking", icon: Building2 },
]

const getServiceFee = (serviceType: string | null) => {
  if (serviceType?.includes("tatkaal")) {
    return { base: 1500, tatkaal: 2000, total: 3500 }
  }
  return { base: 1500, tatkaal: 0, total: 1500 }
}

export function Payment() {
  const { bookingData, setCurrentStep, completePayment } = useBooking()
  const [paymentMethod, setPaymentMethod] = useState("card")
  const [isProcessing, setIsProcessing] = useState(false)
  const [cardNumber, setCardNumber] = useState("")
  const [cardExpiry, setCardExpiry] = useState("")
  const [cardCvv, setCardCvv] = useState("")
  const [upiId, setUpiId] = useState("")

  const fees = getServiceFee(bookingData.serviceType)

  const formatCardNumber = (value: string) => {
    const v = value.replace(/\s+/g, "").replace(/[^0-9]/gi, "")
    const matches = v.match(/\d{4,16}/g)
    const match = (matches && matches[0]) || ""
    const parts = []
    for (let i = 0, len = match.length; i < len; i += 4) {
      parts.push(match.substring(i, i + 4))
    }
    return parts.length ? parts.join(" ") : value
  }

  const formatExpiry = (value: string) => {
    const v = value.replace(/\s+/g, "").replace(/[^0-9]/gi, "")
    if (v.length >= 2) {
      return `${v.substring(0, 2)}/${v.substring(2, 4)}`
    }
    return v
  }

  const handlePayment = async () => {
    setIsProcessing(true)
    // Simulate payment processing
    await new Promise((resolve) => setTimeout(resolve, 2000))
    
    // Generate booking reference
    const reference = `PSP${Date.now().toString(36).toUpperCase()}${Math.random().toString(36).substring(2, 6).toUpperCase()}`
    completePayment(reference)
    setCurrentStep(5)
  }

  const handleBack = () => {
    setCurrentStep(3)
  }

  const canSubmit = () => {
    if (paymentMethod === "card") {
      return cardNumber.replace(/\s/g, "").length === 16 && cardExpiry.length === 5 && cardCvv.length === 3
    }
    if (paymentMethod === "upi") {
      return upiId.includes("@")
    }
    return true
  }

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-2xl font-semibold text-foreground text-balance">
          Complete your payment
        </h2>
        <p className="mt-2 text-muted-foreground">
          Review your appointment details and pay securely to confirm your booking.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Payment Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Appointment Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Appointment Summary</CardTitle>
              <CardDescription>
                Please verify these details before payment
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="flex items-start gap-3">
                  <User className="h-5 w-5 text-primary mt-0.5" aria-hidden="true" />
                  <div>
                    <p className="text-sm text-muted-foreground">Applicant</p>
                    <p className="font-medium">{bookingData.applicant.fullName || "Not provided"}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Shield className="h-5 w-5 text-primary mt-0.5" aria-hidden="true" />
                  <div>
                    <p className="text-sm text-muted-foreground">Service Type</p>
                    <p className="font-medium capitalize">
                      {bookingData.serviceType?.replace("-", " ") || "Not selected"}
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Building2 className="h-5 w-5 text-primary mt-0.5" aria-hidden="true" />
                  <div>
                    <p className="text-sm text-muted-foreground">Location</p>
                    <p className="font-medium">{bookingData.slot?.pskName || "Not selected"}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Calendar className="h-5 w-5 text-primary mt-0.5" aria-hidden="true" />
                  <div>
                    <p className="text-sm text-muted-foreground">Date & Time</p>
                    <p className="font-medium">
                      {bookingData.slot
                        ? `${new Date(bookingData.slot.date).toLocaleDateString("en-IN", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                          })}, ${bookingData.slot.time}`
                        : "Not selected"}
                    </p>
                  </div>
                </div>
              </div>
              <Button variant="link" className="h-auto p-0 text-sm" onClick={() => setCurrentStep(1)}>
                Edit appointment details
              </Button>
            </CardContent>
          </Card>

          {/* Payment Method */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Payment Method</CardTitle>
              <CardDescription>
                Choose your preferred payment method
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <RadioGroup
                value={paymentMethod}
                onValueChange={setPaymentMethod}
                className="grid grid-cols-3 gap-3"
              >
                {paymentMethods.map((method) => {
                  const Icon = method.icon
                  return (
                    <Label
                      key={method.id}
                      htmlFor={method.id}
                      className="cursor-pointer"
                    >
                      <div
                        className={cn(
                          "flex flex-col items-center gap-2 rounded-lg border p-4 transition-all hover:border-primary/50",
                          paymentMethod === method.id && "border-primary bg-primary/5"
                        )}
                      >
                        <RadioGroupItem value={method.id} id={method.id} className="sr-only" />
                        <Icon className="h-6 w-6 text-primary" aria-hidden="true" />
                        <span className="text-sm font-medium text-center">{method.label}</span>
                      </div>
                    </Label>
                  )
                })}
              </RadioGroup>

              {/* Card Details */}
              {paymentMethod === "card" && (
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label htmlFor="cardNumber">Card Number</Label>
                    <Input
                      id="cardNumber"
                      placeholder="1234 5678 9012 3456"
                      value={cardNumber}
                      onChange={(e) => setCardNumber(formatCardNumber(e.target.value))}
                      maxLength={19}
                      className="h-11 text-base"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="expiry">Expiry Date</Label>
                      <Input
                        id="expiry"
                        placeholder="MM/YY"
                        value={cardExpiry}
                        onChange={(e) => setCardExpiry(formatExpiry(e.target.value))}
                        maxLength={5}
                        className="h-11 text-base"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cvv">CVV</Label>
                      <Input
                        id="cvv"
                        placeholder="123"
                        type="password"
                        value={cardCvv}
                        onChange={(e) => setCardCvv(e.target.value.replace(/\D/g, "").slice(0, 3))}
                        maxLength={3}
                        className="h-11 text-base"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* UPI */}
              {paymentMethod === "upi" && (
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label htmlFor="upiId">UPI ID</Label>
                    <Input
                      id="upiId"
                      placeholder="yourname@upi"
                      value={upiId}
                      onChange={(e) => setUpiId(e.target.value)}
                      className="h-11 text-base"
                    />
                  </div>
                </div>
              )}

              {/* Net Banking */}
              {paymentMethod === "netbanking" && (
                <div className="pt-4">
                  <p className="text-sm text-muted-foreground">
                    You will be redirected to your bank&apos;s secure payment page after clicking Pay Now.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Payment Summary */}
        <div className="lg:col-span-1">
          <Card className="sticky top-4">
            <CardHeader>
              <CardTitle className="text-lg">Payment Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Application Fee</span>
                  <span>Rs. {fees.base.toLocaleString("en-IN")}</span>
                </div>
                {fees.tatkaal > 0 && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Tatkaal Fee</span>
                    <span>Rs. {fees.tatkaal.toLocaleString("en-IN")}</span>
                  </div>
                )}
                <Separator />
                <div className="flex justify-between font-semibold">
                  <span>Total Amount</span>
                  <span className="text-lg">Rs. {fees.total.toLocaleString("en-IN")}</span>
                </div>
              </div>

              <Button
                size="lg"
                className="w-full gap-2"
                onClick={handlePayment}
                disabled={isProcessing || !canSubmit()}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Lock className="h-4 w-4" />
                    Pay Rs. {fees.total.toLocaleString("en-IN")}
                  </>
                )}
              </Button>

              <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
                <Lock className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Secured by 256-bit SSL encryption</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Back button */}
      <div className="pt-4">
        <Button
          variant="outline"
          size="lg"
          onClick={handleBack}
          className="gap-2 bg-transparent"
          disabled={isProcessing}
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back to Slot Selection
        </Button>
      </div>
    </div>
  )
}
