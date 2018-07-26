; Auto-generated. Do not edit!


(cl:in-package candy-srv)


;//! \htmlinclude Value-request.msg.html

(cl:defclass <Value-request> (roslisp-msg-protocol:ros-message)
  ((a
    :reader a
    :initarg :a
    :type cl:string
    :initform ""))
)

(cl:defclass Value-request (<Value-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Value-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Value-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<Value-request> is deprecated: use candy-srv:Value-request instead.")))

(cl:ensure-generic-function 'a-val :lambda-list '(m))
(cl:defmethod a-val ((m <Value-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:a-val is deprecated.  Use candy-srv:a instead.")
  (a m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Value-request>) ostream)
  "Serializes a message object of type '<Value-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'a))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'a))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Value-request>) istream)
  "Deserializes a message object of type '<Value-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'a) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'a) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Value-request>)))
  "Returns string type for a service object of type '<Value-request>"
  "candy/ValueRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Value-request)))
  "Returns string type for a service object of type 'Value-request"
  "candy/ValueRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Value-request>)))
  "Returns md5sum for a message object of type '<Value-request>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Value-request)))
  "Returns md5sum for a message object of type 'Value-request"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Value-request>)))
  "Returns full string definition for message of type '<Value-request>"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Value-request)))
  "Returns full string definition for message of type 'Value-request"
  (cl:format cl:nil "string a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Value-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'a))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Value-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Value-request
    (cl:cons ':a (a msg))
))
;//! \htmlinclude Value-response.msg.html

(cl:defclass <Value-response> (roslisp-msg-protocol:ros-message)
  ((b
    :reader b
    :initarg :b
    :type cl:string
    :initform ""))
)

(cl:defclass Value-response (<Value-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Value-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Value-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name candy-srv:<Value-response> is deprecated: use candy-srv:Value-response instead.")))

(cl:ensure-generic-function 'b-val :lambda-list '(m))
(cl:defmethod b-val ((m <Value-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader candy-srv:b-val is deprecated.  Use candy-srv:b instead.")
  (b m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Value-response>) ostream)
  "Serializes a message object of type '<Value-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'b))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'b))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Value-response>) istream)
  "Deserializes a message object of type '<Value-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'b) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'b) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Value-response>)))
  "Returns string type for a service object of type '<Value-response>"
  "candy/ValueResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Value-response)))
  "Returns string type for a service object of type 'Value-response"
  "candy/ValueResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Value-response>)))
  "Returns md5sum for a message object of type '<Value-response>"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Value-response)))
  "Returns md5sum for a message object of type 'Value-response"
  "945e963769938e4ddc3288e80fdfddf4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Value-response>)))
  "Returns full string definition for message of type '<Value-response>"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Value-response)))
  "Returns full string definition for message of type 'Value-response"
  (cl:format cl:nil "string b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Value-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'b))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Value-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Value-response
    (cl:cons ':b (b msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Value)))
  'Value-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Value)))
  'Value-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Value)))
  "Returns string type for a service object of type '<Value>"
  "candy/Value")