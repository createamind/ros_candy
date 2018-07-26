;; Auto-generated. Do not edit!


(when (boundp 'candy::Step)
  (if (not (find-package "CANDY"))
    (make-package "CANDY"))
  (shadow 'Step (find-package "CANDY")))
(unless (find-package "CANDY::STEP")
  (make-package "CANDY::STEP"))
(unless (find-package "CANDY::STEPREQUEST")
  (make-package "CANDY::STEPREQUEST"))
(unless (find-package "CANDY::STEPRESPONSE")
  (make-package "CANDY::STEPRESPONSE"))

(in-package "ROS")





(defclass candy::StepRequest
  :super ros::object
  :slots (_a ))

(defmethod candy::StepRequest
  (:init
   (&key
    ((:a __a) "")
    )
   (send-super :init)
   (setq _a (string __a))
   self)
  (:a
   (&optional __a)
   (if __a (setq _a __a)) _a)
  (:serialization-length
   ()
   (+
    ;; string _a
    4 (length _a)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _a
       (write-long (length _a) s) (princ _a s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _a
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _a (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass candy::StepResponse
  :super ros::object
  :slots (_b ))

(defmethod candy::StepResponse
  (:init
   (&key
    ((:b __b) "")
    )
   (send-super :init)
   (setq _b (string __b))
   self)
  (:b
   (&optional __b)
   (if __b (setq _b __b)) _b)
  (:serialization-length
   ()
   (+
    ;; string _b
    4 (length _b)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _b
       (write-long (length _b) s) (princ _b s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _b
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _b (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass candy::Step
  :super ros::object
  :slots ())

(setf (get candy::Step :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::Step :datatype-) "candy/Step")
(setf (get candy::Step :request) candy::StepRequest)
(setf (get candy::Step :response) candy::StepResponse)

(defmethod candy::StepRequest
  (:response () (instance candy::StepResponse :init)))

(setf (get candy::StepRequest :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::StepRequest :datatype-) "candy/StepRequest")
(setf (get candy::StepRequest :definition-)
      "string a
---
string b
")

(setf (get candy::StepResponse :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::StepResponse :datatype-) "candy/StepResponse")
(setf (get candy::StepResponse :definition-)
      "string a
---
string b
")



(provide :candy/Step "945e963769938e4ddc3288e80fdfddf4")


