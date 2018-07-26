;; Auto-generated. Do not edit!


(when (boundp 'candy::Value)
  (if (not (find-package "CANDY"))
    (make-package "CANDY"))
  (shadow 'Value (find-package "CANDY")))
(unless (find-package "CANDY::VALUE")
  (make-package "CANDY::VALUE"))
(unless (find-package "CANDY::VALUEREQUEST")
  (make-package "CANDY::VALUEREQUEST"))
(unless (find-package "CANDY::VALUERESPONSE")
  (make-package "CANDY::VALUERESPONSE"))

(in-package "ROS")





(defclass candy::ValueRequest
  :super ros::object
  :slots (_a ))

(defmethod candy::ValueRequest
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

(defclass candy::ValueResponse
  :super ros::object
  :slots (_b ))

(defmethod candy::ValueResponse
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

(defclass candy::Value
  :super ros::object
  :slots ())

(setf (get candy::Value :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::Value :datatype-) "candy/Value")
(setf (get candy::Value :request) candy::ValueRequest)
(setf (get candy::Value :response) candy::ValueResponse)

(defmethod candy::ValueRequest
  (:response () (instance candy::ValueResponse :init)))

(setf (get candy::ValueRequest :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::ValueRequest :datatype-) "candy/ValueRequest")
(setf (get candy::ValueRequest :definition-)
      "string a
---
string b
")

(setf (get candy::ValueResponse :md5sum-) "945e963769938e4ddc3288e80fdfddf4")
(setf (get candy::ValueResponse :datatype-) "candy/ValueResponse")
(setf (get candy::ValueResponse :definition-)
      "string a
---
string b
")



(provide :candy/Value "945e963769938e4ddc3288e80fdfddf4")


