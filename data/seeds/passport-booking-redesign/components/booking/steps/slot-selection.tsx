"use client"

import { useState } from "react"
import { ArrowLeft, ArrowRight, MapPin, Clock, Calendar, Building2, ChevronLeft, ChevronRight, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { HelpTooltip } from "@/components/booking/help-tooltip"
import { useBooking, type SlotDetails } from "@/lib/booking-context"
import { cn } from "@/lib/utils"

// Mock PSK data
const pskLocations = [
  {
    id: "psk-mumbai-01",
    name: "PSK Mumbai (Malad)",
    address: "Mindspace, Malad West, Mumbai - 400064",
    type: "PSK",
    distance: "3.2 km",
  },
  {
    id: "psk-mumbai-02",
    name: "PSK Mumbai (BKC)",
    address: "Bandra Kurla Complex, Mumbai - 400051",
    type: "PSK",
    distance: "8.5 km",
  },
  {
    id: "popsk-thane-01",
    name: "POPSK Thane",
    address: "Post Office Complex, Thane West - 400601",
    type: "POPSK",
    distance: "12.1 km",
  },
]

// Generate available slots for a date
const generateTimeSlots = (date: Date) => {
  const slots = []
  const startHour = 9
  const endHour = 17
  
  for (let hour = startHour; hour < endHour; hour++) {
    const available = Math.random() > 0.3 // 70% chance of availability
    const remaining = available ? Math.floor(Math.random() * 15) + 1 : 0
    slots.push({
      id: `slot-${hour}`,
      time: `${hour.toString().padStart(2, "0")}:00 - ${(hour + 1).toString().padStart(2, "0")}:00`,
      available,
      remaining,
    })
  }
  return slots
}

// Get next 14 days
const getAvailableDates = () => {
  const dates = []
  const today = new Date()
  for (let i = 3; i < 17; i++) { // Start from 3 days ahead
    const date = new Date(today)
    date.setDate(today.getDate() + i)
    dates.push(date)
  }
  return dates
}

export function SlotSelection() {
  const { bookingData, updateSlotDetails, setCurrentStep } = useBooking()
  const [selectedPsk, setSelectedPsk] = useState<string>(bookingData.slot?.pskId || "")
  const [selectedDate, setSelectedDate] = useState<Date | null>(
    bookingData.slot?.date ? new Date(bookingData.slot.date) : null
  )
  const [selectedSlot, setSelectedSlot] = useState<string>(bookingData.slot?.slotId || "")
  const [dateOffset, setDateOffset] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  const availableDates = getAvailableDates()
  const visibleDates = availableDates.slice(dateOffset, dateOffset + 7)
  const timeSlots = selectedDate ? generateTimeSlots(selectedDate) : []

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-IN", {
      weekday: "short",
      day: "numeric",
      month: "short",
    })
  }

  const handlePskSelect = (pskId: string) => {
    setSelectedPsk(pskId)
    setSelectedDate(null)
    setSelectedSlot("")
    // Simulate loading slots
    setIsLoading(true)
    setTimeout(() => setIsLoading(false), 500)
  }

  const handleDateSelect = (date: Date) => {
    setSelectedDate(date)
    setSelectedSlot("")
    setIsLoading(true)
    setTimeout(() => setIsLoading(false), 300)
  }

  const handleSlotSelect = (slotId: string) => {
    setSelectedSlot(slotId)
    
    const psk = pskLocations.find(p => p.id === selectedPsk)
    const slot = timeSlots.find(s => s.id === slotId)
    
    if (psk && selectedDate && slot) {
      const slotDetails: SlotDetails = {
        pskId: psk.id,
        pskName: psk.name,
        pskAddress: psk.address,
        date: selectedDate.toISOString(),
        time: slot.time,
        slotId: slot.id,
      }
      updateSlotDetails(slotDetails)
    }
  }

  const handleContinue = () => {
    if (selectedPsk && selectedDate && selectedSlot) {
      setCurrentStep(4)
    }
  }

  const handleBack = () => {
    setCurrentStep(2)
  }

  const selectedPskData = pskLocations.find(p => p.id === selectedPsk)

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-2xl font-semibold text-foreground text-balance">
          Choose your appointment location and time
        </h2>
        <p className="mt-2 text-muted-foreground">
          Select a Passport Seva Kendra near you and pick a convenient appointment slot.
        </p>
      </div>

      {/* PSK Selection */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Building2 className="h-5 w-5 text-primary" aria-hidden="true" />
                Select Passport Seva Kendra
              </CardTitle>
              <CardDescription className="mt-1">
                Choose a location nearest to your address
              </CardDescription>
            </div>
            <HelpTooltip
              title="PSK vs POPSK"
              content="PSK (Passport Seva Kendra) handles all services. POPSK (Post Office PSK) is available at select post offices for limited services."
            />
          </div>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={selectedPsk}
            onValueChange={handlePskSelect}
            className="space-y-3"
          >
            {pskLocations.map((psk) => (
              <Label
                key={psk.id}
                htmlFor={psk.id}
                className="cursor-pointer"
              >
                <div
                  className={cn(
                    "flex items-start gap-4 rounded-lg border p-4 transition-all hover:border-primary/50",
                    selectedPsk === psk.id && "border-primary bg-primary/5"
                  )}
                >
                  <RadioGroupItem value={psk.id} id={psk.id} className="mt-1" />
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-foreground">{psk.name}</span>
                      <Badge variant={psk.type === "PSK" ? "default" : "secondary"} className="text-xs">
                        {psk.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground flex items-center gap-1">
                      <MapPin className="h-3.5 w-3.5" aria-hidden="true" />
                      {psk.address}
                    </p>
                  </div>
                  <span className="text-sm text-muted-foreground">{psk.distance}</span>
                </div>
              </Label>
            ))}
          </RadioGroup>
        </CardContent>
      </Card>

      {/* Date Selection */}
      {selectedPsk && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Calendar className="h-5 w-5 text-primary" aria-hidden="true" />
              Select Appointment Date
            </CardTitle>
            <CardDescription>
              Showing available dates for {selectedPskData?.name}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Date navigation */}
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setDateOffset(Math.max(0, dateOffset - 7))}
                disabled={dateOffset === 0}
                aria-label="Previous week"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              
              <div className="flex-1 grid grid-cols-7 gap-2">
                {visibleDates.map((date) => {
                  const isSelected = selectedDate?.toDateString() === date.toDateString()
                  const isWeekend = date.getDay() === 0 // Sunday
                  
                  return (
                    <button
                      key={date.toISOString()}
                      onClick={() => !isWeekend && handleDateSelect(date)}
                      disabled={isWeekend}
                      className={cn(
                        "flex flex-col items-center rounded-lg border p-3 text-sm transition-all",
                        isSelected && "border-primary bg-primary text-primary-foreground",
                        !isSelected && !isWeekend && "hover:border-primary/50",
                        isWeekend && "opacity-50 cursor-not-allowed bg-muted"
                      )}
                    >
                      <span className="text-xs font-medium">
                        {date.toLocaleDateString("en-IN", { weekday: "short" })}
                      </span>
                      <span className="text-lg font-semibold">{date.getDate()}</span>
                      <span className="text-xs">
                        {date.toLocaleDateString("en-IN", { month: "short" })}
                      </span>
                    </button>
                  )
                })}
              </div>

              <Button
                variant="outline"
                size="icon"
                onClick={() => setDateOffset(Math.min(availableDates.length - 7, dateOffset + 7))}
                disabled={dateOffset >= availableDates.length - 7}
                aria-label="Next week"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Time Slot Selection */}
      {selectedDate && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Clock className="h-5 w-5 text-primary" aria-hidden="true" />
              Select Time Slot
            </CardTitle>
            <CardDescription>
              Available slots for {formatDate(selectedDate)}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                <span className="ml-2 text-muted-foreground">Loading available slots...</span>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {timeSlots.map((slot) => (
                  <button
                    key={slot.id}
                    onClick={() => slot.available && handleSlotSelect(slot.id)}
                    disabled={!slot.available}
                    className={cn(
                      "flex flex-col items-center rounded-lg border p-4 transition-all",
                      selectedSlot === slot.id && "border-primary bg-primary text-primary-foreground",
                      slot.available && selectedSlot !== slot.id && "hover:border-primary/50",
                      !slot.available && "opacity-50 cursor-not-allowed bg-muted"
                    )}
                  >
                    <span className="font-medium">{slot.time}</span>
                    {slot.available ? (
                      <span className={cn(
                        "mt-1 text-xs",
                        selectedSlot === slot.id ? "text-primary-foreground/80" : "text-green-600"
                      )}>
                        {slot.remaining} slots left
                      </span>
                    ) : (
                      <span className="mt-1 text-xs text-muted-foreground">Fully booked</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Selection Summary */}
      {selectedSlot && bookingData.slot && (
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="py-4">
            <h3 className="font-medium text-foreground mb-3">Your Selection</h3>
            <div className="grid gap-2 text-sm">
              <div className="flex items-center gap-2">
                <Building2 className="h-4 w-4 text-primary" aria-hidden="true" />
                <span>{bookingData.slot.pskName}</span>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4 text-primary" aria-hidden="true" />
                <span>{new Date(bookingData.slot.date).toLocaleDateString("en-IN", {
                  weekday: "long",
                  day: "numeric",
                  month: "long",
                  year: "numeric"
                })}</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-primary" aria-hidden="true" />
                <span>{bookingData.slot.time}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

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
          disabled={!selectedSlot}
          className="gap-2 px-8"
        >
          Continue to Payment
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  )
}
