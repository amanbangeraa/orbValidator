"use client"

import { CheckCircle2, Download, Printer, Calendar, Clock, Building2, User, FileText, Phone, Mail, MapPin, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useBooking } from "@/lib/booking-context"

const documentsRequired = [
  "Original and self-attested copy of proof of date of birth",
  "Original and self-attested copy of address proof",
  "Original and self-attested copy of ID proof (Aadhaar/PAN/Voter ID)",
  "Two recent passport-size photographs (white background)",
  "Old passport (for reissue applications)",
]

export function Confirmation() {
  const { bookingData, resetBooking } = useBooking()

  const handlePrint = () => {
    window.print()
  }

  const handleDownload = () => {
    // In a real app, this would generate a PDF
    alert("Download functionality would generate a PDF receipt")
  }

  const handleNewBooking = () => {
    resetBooking()
  }

  return (
    <div className="space-y-8">
      {/* Success Header */}
      <div className="text-center space-y-4">
        <div className="flex justify-center">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-green-100">
            <CheckCircle2 className="h-12 w-12 text-green-600" aria-hidden="true" />
          </div>
        </div>
        <div>
          <h2 className="text-2xl font-semibold text-foreground">
            Appointment Successfully Booked
          </h2>
          <p className="mt-2 text-muted-foreground">
            Your passport appointment has been confirmed. Please save this confirmation for your records.
          </p>
        </div>
      </div>

      {/* Booking Reference */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="py-6 text-center">
          <p className="text-sm text-muted-foreground">Booking Reference Number</p>
          <p className="mt-1 text-3xl font-bold tracking-wider text-primary">
            {bookingData.bookingReference}
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            Please note down this reference number for future correspondence
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Appointment Details */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Appointment Details</CardTitle>
            <CardDescription>
              Your scheduled passport appointment
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start gap-3">
              <Calendar className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Date</p>
                <p className="font-medium">
                  {bookingData.slot
                    ? new Date(bookingData.slot.date).toLocaleDateString("en-IN", {
                        weekday: "long",
                        day: "numeric",
                        month: "long",
                        year: "numeric",
                      })
                    : "Not available"}
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <Clock className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Time</p>
                <p className="font-medium">{bookingData.slot?.time || "Not available"}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Building2 className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Location</p>
                <p className="font-medium">{bookingData.slot?.pskName || "Not available"}</p>
                <p className="text-sm text-muted-foreground mt-0.5">{bookingData.slot?.pskAddress}</p>
              </div>
            </div>

            <Separator />

            <div className="flex items-start gap-3">
              <User className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Applicant</p>
                <p className="font-medium">{bookingData.applicant.fullName}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <FileText className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Service Type</p>
                <p className="font-medium capitalize">
                  {bookingData.serviceType?.replace("-", " ") || "Not selected"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Applicant Contact */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Contact Information</CardTitle>
            <CardDescription>
              Confirmation sent to these details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start gap-3">
              <Mail className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Email</p>
                <p className="font-medium">{bookingData.applicant.email}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Phone className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Mobile</p>
                <p className="font-medium">{bookingData.applicant.phone}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <MapPin className="h-5 w-5 text-primary mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="text-sm text-muted-foreground">Address</p>
                <p className="font-medium">
                  {bookingData.applicant.address}, {bookingData.applicant.city}
                </p>
                <p className="text-sm text-muted-foreground">
                  {bookingData.applicant.state} - {bookingData.applicant.pincode}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Documents Checklist */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <FileText className="h-5 w-5 text-primary" aria-hidden="true" />
            Documents to Bring
          </CardTitle>
          <CardDescription>
            Please bring the following original documents to your appointment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {documentsRequired.map((doc, index) => (
              <li key={index} className="flex items-start gap-3">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
                  {index + 1}
                </div>
                <span className="text-sm">{doc}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Important Notes */}
      <Card className="border-accent/30 bg-accent/5">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-accent mt-0.5 shrink-0" aria-hidden="true" />
            <div className="space-y-2">
              <h3 className="font-medium text-foreground">Important Information</h3>
              <ul className="text-sm text-muted-foreground space-y-1.5">
                <li>Please arrive 15 minutes before your scheduled appointment time.</li>
                <li>You can reschedule your appointment up to 3 days before the scheduled date.</li>
                <li>Keep a printed copy of this confirmation along with your documents.</li>
                <li>For any queries, call our helpline: 1800-258-1800 (8 AM - 10 PM)</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
        <Button
          variant="outline"
          size="lg"
          onClick={handleDownload}
          className="gap-2 w-full sm:w-auto bg-transparent"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          Download Receipt
        </Button>
        <Button
          variant="outline"
          size="lg"
          onClick={handlePrint}
          className="gap-2 w-full sm:w-auto bg-transparent"
        >
          <Printer className="h-4 w-4" aria-hidden="true" />
          Print Confirmation
        </Button>
        <Button
          size="lg"
          onClick={handleNewBooking}
          className="gap-2 w-full sm:w-auto"
        >
          Book Another Appointment
        </Button>
      </div>
    </div>
  )
}
