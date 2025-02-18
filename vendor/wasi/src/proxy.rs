use crate::io::poll as __with_name0;
use crate::clocks::monotonic_clock as __with_name1;
use crate::clocks::wall_clock as __with_name2;
use crate::random::random as __with_name3;
use crate::io::error as __with_name4;
use crate::io::streams as __with_name5;
use crate::cli::stdout as __with_name6;
use crate::cli::stderr as __with_name7;
use crate::cli::stdin as __with_name8;
use crate::http::types as __with_name9;
use crate::http::outgoing_handler as __with_name10;
#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod wasi {
        #[allow(dead_code)]
        pub mod http {
            #[allow(dead_code, clippy::all)]
            pub mod incoming_handler {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                pub type IncomingRequest = super::super::super::super::__with_name9::IncomingRequest;
                pub type ResponseOutparam = super::super::super::super::__with_name9::ResponseOutparam;
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_handle_cabi<T: Guest>(arg0: i32, arg1: i32) {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    T::handle(
                        super::super::super::super::__with_name9::IncomingRequest::from_handle(
                            arg0 as u32,
                        ),
                        super::super::super::super::__with_name9::ResponseOutparam::from_handle(
                            arg1 as u32,
                        ),
                    );
                }
                pub trait Guest {
                    /// This function is invoked with an incoming HTTP Request, and a resource
                    /// `response-outparam` which provides the capability to reply with an HTTP
                    /// Response. The response is sent by calling the `response-outparam.set`
                    /// method, which allows execution to continue after the response has been
                    /// sent. This enables both streaming to the response body, and performing other
                    /// work.
                    ///
                    /// The implementor of this function must write a response to the
                    /// `response-outparam` before returning, or else the caller will respond
                    /// with an error on its behalf.
                    fn handle(request: IncomingRequest, response_out: ResponseOutparam);
                }
                #[doc(hidden)]
                #[macro_export]
                macro_rules! __export_wasi_http_incoming_handler_0_2_2_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "wasi:http/incoming-handler@0.2.2#handle"] unsafe extern "C" fn
                        export_handle(arg0 : i32, arg1 : i32,) { $($path_to_types)*::
                        _export_handle_cabi::<$ty > (arg0, arg1) } };
                    };
                }
                #[doc(hidden)]
                pub use __export_wasi_http_incoming_handler_0_2_2_cabi;
            }
        }
    }
}
mod _rt {
    #[cfg(target_arch = "wasm32")]
    pub fn run_ctors_once() {
        wit_bindgen_rt::run_ctors_once();
    }
}
/// Generates `#[no_mangle]` functions to export the specified type as the
/// root implementation of all generated traits.
///
/// For more information see the documentation of `wit_bindgen::generate!`.
///
/// ```rust
/// # macro_rules! _export_proxy{ ($($t:tt)*) => (); }
/// # trait Guest {}
/// struct MyType;
///
/// impl Guest for MyType {
///     // ...
/// }
///
/// _export_proxy!(MyType);
/// ```
#[allow(unused_macros)]
#[doc(hidden)]
#[macro_export]
macro_rules! __export_proxy_impl {
    ($ty:ident) => {
        wasi::_export_proxy!($ty with_types_in wasi);
    };
    ($ty:ident with_types_in $($path_to_types_root:tt)*) => {
        $($path_to_types_root)*::
        exports::wasi::http::incoming_handler::__export_wasi_http_incoming_handler_0_2_2_cabi!($ty
        with_types_in $($path_to_types_root)*:: exports::wasi::http::incoming_handler);
        const _ : () = { #[cfg(target_arch = "wasm32")] #[link_section =
        "component-type:wit-bindgen:0.33.0:wasi:http@0.2.2:proxy:imports and exportsrust-wasi-from-crates-io-proxy-world"]
        #[doc(hidden)] pub static __WIT_BINDGEN_COMPONENT_TYPE : [u8; 7040] = *
        b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\x846\x01A\x02\x01A#\x01\
B\x0a\x04\0\x08pollable\x03\x01\x01h\0\x01@\x01\x04self\x01\0\x7f\x04\0\x16[meth\
od]pollable.ready\x01\x02\x01@\x01\x04self\x01\x01\0\x04\0\x16[method]pollable.b\
lock\x01\x03\x01p\x01\x01py\x01@\x01\x02in\x04\0\x05\x04\0\x04poll\x01\x06\x03\0\
\x12wasi:io/poll@0.2.2\x05\0\x02\x03\0\0\x08pollable\x01B\x0f\x02\x03\x02\x01\x01\
\x04\0\x08pollable\x03\0\0\x01w\x04\0\x07instant\x03\0\x02\x01w\x04\0\x08duratio\
n\x03\0\x04\x01@\0\0\x03\x04\0\x03now\x01\x06\x01@\0\0\x05\x04\0\x0aresolution\x01\
\x07\x01i\x01\x01@\x01\x04when\x03\0\x08\x04\0\x11subscribe-instant\x01\x09\x01@\
\x01\x04when\x05\0\x08\x04\0\x12subscribe-duration\x01\x0a\x03\0!wasi:clocks/mon\
otonic-clock@0.2.2\x05\x02\x01B\x05\x01r\x02\x07secondsw\x0bnanosecondsy\x04\0\x08\
datetime\x03\0\0\x01@\0\0\x01\x04\0\x03now\x01\x02\x04\0\x0aresolution\x01\x02\x03\
\0\x1cwasi:clocks/wall-clock@0.2.2\x05\x03\x01B\x05\x01p}\x01@\x01\x03lenw\0\0\x04\
\0\x10get-random-bytes\x01\x01\x01@\0\0w\x04\0\x0eget-random-u64\x01\x02\x03\0\x18\
wasi:random/random@0.2.2\x05\x04\x01B\x04\x04\0\x05error\x03\x01\x01h\0\x01@\x01\
\x04self\x01\0s\x04\0\x1d[method]error.to-debug-string\x01\x02\x03\0\x13wasi:io/\
error@0.2.2\x05\x05\x02\x03\0\x04\x05error\x01B(\x02\x03\x02\x01\x06\x04\0\x05er\
ror\x03\0\0\x02\x03\x02\x01\x01\x04\0\x08pollable\x03\0\x02\x01i\x01\x01q\x02\x15\
last-operation-failed\x01\x04\0\x06closed\0\0\x04\0\x0cstream-error\x03\0\x05\x04\
\0\x0cinput-stream\x03\x01\x04\0\x0doutput-stream\x03\x01\x01h\x07\x01p}\x01j\x01\
\x0a\x01\x06\x01@\x02\x04self\x09\x03lenw\0\x0b\x04\0\x19[method]input-stream.re\
ad\x01\x0c\x04\0\"[method]input-stream.blocking-read\x01\x0c\x01j\x01w\x01\x06\x01\
@\x02\x04self\x09\x03lenw\0\x0d\x04\0\x19[method]input-stream.skip\x01\x0e\x04\0\
\"[method]input-stream.blocking-skip\x01\x0e\x01i\x03\x01@\x01\x04self\x09\0\x0f\
\x04\0\x1e[method]input-stream.subscribe\x01\x10\x01h\x08\x01@\x01\x04self\x11\0\
\x0d\x04\0![method]output-stream.check-write\x01\x12\x01j\0\x01\x06\x01@\x02\x04\
self\x11\x08contents\x0a\0\x13\x04\0\x1b[method]output-stream.write\x01\x14\x04\0\
.[method]output-stream.blocking-write-and-flush\x01\x14\x01@\x01\x04self\x11\0\x13\
\x04\0\x1b[method]output-stream.flush\x01\x15\x04\0$[method]output-stream.blocki\
ng-flush\x01\x15\x01@\x01\x04self\x11\0\x0f\x04\0\x1f[method]output-stream.subsc\
ribe\x01\x16\x01@\x02\x04self\x11\x03lenw\0\x13\x04\0\"[method]output-stream.wri\
te-zeroes\x01\x17\x04\05[method]output-stream.blocking-write-zeroes-and-flush\x01\
\x17\x01@\x03\x04self\x11\x03src\x09\x03lenw\0\x0d\x04\0\x1c[method]output-strea\
m.splice\x01\x18\x04\0%[method]output-stream.blocking-splice\x01\x18\x03\0\x15wa\
si:io/streams@0.2.2\x05\x07\x02\x03\0\x05\x0doutput-stream\x01B\x05\x02\x03\x02\x01\
\x08\x04\0\x0doutput-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x0aget-stdout\x01\
\x03\x03\0\x15wasi:cli/stdout@0.2.2\x05\x09\x01B\x05\x02\x03\x02\x01\x08\x04\0\x0d\
output-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x0aget-stderr\x01\x03\x03\0\x15\
wasi:cli/stderr@0.2.2\x05\x0a\x02\x03\0\x05\x0cinput-stream\x01B\x05\x02\x03\x02\
\x01\x0b\x04\0\x0cinput-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x09get-stdin\x01\
\x03\x03\0\x14wasi:cli/stdin@0.2.2\x05\x0c\x02\x03\0\x01\x08duration\x01B\xc1\x01\
\x02\x03\x02\x01\x0d\x04\0\x08duration\x03\0\0\x02\x03\x02\x01\x0b\x04\0\x0cinpu\
t-stream\x03\0\x02\x02\x03\x02\x01\x08\x04\0\x0doutput-stream\x03\0\x04\x02\x03\x02\
\x01\x06\x04\0\x08io-error\x03\0\x06\x02\x03\x02\x01\x01\x04\0\x08pollable\x03\0\
\x08\x01q\x0a\x03get\0\0\x04head\0\0\x04post\0\0\x03put\0\0\x06delete\0\0\x07con\
nect\0\0\x07options\0\0\x05trace\0\0\x05patch\0\0\x05other\x01s\0\x04\0\x06metho\
d\x03\0\x0a\x01q\x03\x04HTTP\0\0\x05HTTPS\0\0\x05other\x01s\0\x04\0\x06scheme\x03\
\0\x0c\x01ks\x01k{\x01r\x02\x05rcode\x0e\x09info-code\x0f\x04\0\x11DNS-error-pay\
load\x03\0\x10\x01k}\x01r\x02\x08alert-id\x12\x0dalert-message\x0e\x04\0\x1aTLS-\
alert-received-payload\x03\0\x13\x01ky\x01r\x02\x0afield-name\x0e\x0afield-size\x15\
\x04\0\x12field-size-payload\x03\0\x16\x01kw\x01k\x17\x01q'\x0bDNS-timeout\0\0\x09\
DNS-error\x01\x11\0\x15destination-not-found\0\0\x17destination-unavailable\0\0\x19\
destination-IP-prohibited\0\0\x19destination-IP-unroutable\0\0\x12connection-ref\
used\0\0\x15connection-terminated\0\0\x12connection-timeout\0\0\x17connection-re\
ad-timeout\0\0\x18connection-write-timeout\0\0\x18connection-limit-reached\0\0\x12\
TLS-protocol-error\0\0\x15TLS-certificate-error\0\0\x12TLS-alert-received\x01\x14\
\0\x13HTTP-request-denied\0\0\x1cHTTP-request-length-required\0\0\x16HTTP-reques\
t-body-size\x01\x18\0\x1bHTTP-request-method-invalid\0\0\x18HTTP-request-URI-inv\
alid\0\0\x19HTTP-request-URI-too-long\0\0\x20HTTP-request-header-section-size\x01\
\x15\0\x18HTTP-request-header-size\x01\x19\0!HTTP-request-trailer-section-size\x01\
\x15\0\x19HTTP-request-trailer-size\x01\x17\0\x18HTTP-response-incomplete\0\0!HT\
TP-response-header-section-size\x01\x15\0\x19HTTP-response-header-size\x01\x17\0\
\x17HTTP-response-body-size\x01\x18\0\"HTTP-response-trailer-section-size\x01\x15\
\0\x1aHTTP-response-trailer-size\x01\x17\0\x1dHTTP-response-transfer-coding\x01\x0e\
\0\x1cHTTP-response-content-coding\x01\x0e\0\x15HTTP-response-timeout\0\0\x13HTT\
P-upgrade-failed\0\0\x13HTTP-protocol-error\0\0\x0dloop-detected\0\0\x13configur\
ation-error\0\0\x0einternal-error\x01\x0e\0\x04\0\x0aerror-code\x03\0\x1a\x01q\x03\
\x0einvalid-syntax\0\0\x09forbidden\0\0\x09immutable\0\0\x04\0\x0cheader-error\x03\
\0\x1c\x01s\x04\0\x09field-key\x03\0\x1e\x04\0\x0afield-name\x03\0\x1f\x01p}\x04\
\0\x0bfield-value\x03\0!\x04\0\x06fields\x03\x01\x04\0\x07headers\x03\0#\x04\0\x08\
trailers\x03\0#\x04\0\x10incoming-request\x03\x01\x04\0\x10outgoing-request\x03\x01\
\x04\0\x0frequest-options\x03\x01\x04\0\x11response-outparam\x03\x01\x01{\x04\0\x0b\
status-code\x03\0*\x04\0\x11incoming-response\x03\x01\x04\0\x0dincoming-body\x03\
\x01\x04\0\x0ffuture-trailers\x03\x01\x04\0\x11outgoing-response\x03\x01\x04\0\x0d\
outgoing-body\x03\x01\x04\0\x18future-incoming-response\x03\x01\x01i#\x01@\0\02\x04\
\0\x13[constructor]fields\x013\x01o\x02\x20\"\x01p4\x01j\x012\x01\x1d\x01@\x01\x07\
entries5\06\x04\0\x18[static]fields.from-list\x017\x01h#\x01p\"\x01@\x02\x04self\
8\x04name\x20\09\x04\0\x12[method]fields.get\x01:\x01@\x02\x04self8\x04name\x20\0\
\x7f\x04\0\x12[method]fields.has\x01;\x01j\0\x01\x1d\x01@\x03\x04self8\x04name\x20\
\x05value9\0<\x04\0\x12[method]fields.set\x01=\x01@\x02\x04self8\x04name\x20\0<\x04\
\0\x15[method]fields.delete\x01>\x01@\x03\x04self8\x04name\x20\x05value\"\0<\x04\
\0\x15[method]fields.append\x01?\x01@\x01\x04self8\05\x04\0\x16[method]fields.en\
tries\x01@\x01@\x01\x04self8\02\x04\0\x14[method]fields.clone\x01A\x01h&\x01@\x01\
\x04self\xc2\0\0\x0b\x04\0\x1f[method]incoming-request.method\x01C\x01@\x01\x04s\
elf\xc2\0\0\x0e\x04\0([method]incoming-request.path-with-query\x01D\x01k\x0d\x01\
@\x01\x04self\xc2\0\0\xc5\0\x04\0\x1f[method]incoming-request.scheme\x01F\x04\0\"\
[method]incoming-request.authority\x01D\x01i$\x01@\x01\x04self\xc2\0\0\xc7\0\x04\
\0\x20[method]incoming-request.headers\x01H\x01i-\x01j\x01\xc9\0\0\x01@\x01\x04s\
elf\xc2\0\0\xca\0\x04\0\x20[method]incoming-request.consume\x01K\x01i'\x01@\x01\x07\
headers\xc7\0\0\xcc\0\x04\0\x1d[constructor]outgoing-request\x01M\x01h'\x01i0\x01\
j\x01\xcf\0\0\x01@\x01\x04self\xce\0\0\xd0\0\x04\0\x1d[method]outgoing-request.b\
ody\x01Q\x01@\x01\x04self\xce\0\0\x0b\x04\0\x1f[method]outgoing-request.method\x01\
R\x01j\0\0\x01@\x02\x04self\xce\0\x06method\x0b\0\xd3\0\x04\0#[method]outgoing-r\
equest.set-method\x01T\x01@\x01\x04self\xce\0\0\x0e\x04\0([method]outgoing-reque\
st.path-with-query\x01U\x01@\x02\x04self\xce\0\x0fpath-with-query\x0e\0\xd3\0\x04\
\0,[method]outgoing-request.set-path-with-query\x01V\x01@\x01\x04self\xce\0\0\xc5\
\0\x04\0\x1f[method]outgoing-request.scheme\x01W\x01@\x02\x04self\xce\0\x06schem\
e\xc5\0\0\xd3\0\x04\0#[method]outgoing-request.set-scheme\x01X\x04\0\"[method]ou\
tgoing-request.authority\x01U\x01@\x02\x04self\xce\0\x09authority\x0e\0\xd3\0\x04\
\0&[method]outgoing-request.set-authority\x01Y\x01@\x01\x04self\xce\0\0\xc7\0\x04\
\0\x20[method]outgoing-request.headers\x01Z\x01i(\x01@\0\0\xdb\0\x04\0\x1c[const\
ructor]request-options\x01\\\x01h(\x01k\x01\x01@\x01\x04self\xdd\0\0\xde\0\x04\0\
'[method]request-options.connect-timeout\x01_\x01@\x02\x04self\xdd\0\x08duration\
\xde\0\0\xd3\0\x04\0+[method]request-options.set-connect-timeout\x01`\x04\0*[met\
hod]request-options.first-byte-timeout\x01_\x04\0.[method]request-options.set-fi\
rst-byte-timeout\x01`\x04\0-[method]request-options.between-bytes-timeout\x01_\x04\
\01[method]request-options.set-between-bytes-timeout\x01`\x01i)\x01i/\x01j\x01\xe2\
\0\x01\x1b\x01@\x02\x05param\xe1\0\x08response\xe3\0\x01\0\x04\0\x1d[static]resp\
onse-outparam.set\x01d\x01h,\x01@\x01\x04self\xe5\0\0+\x04\0\x20[method]incoming\
-response.status\x01f\x01@\x01\x04self\xe5\0\0\xc7\0\x04\0![method]incoming-resp\
onse.headers\x01g\x01@\x01\x04self\xe5\0\0\xca\0\x04\0![method]incoming-response\
.consume\x01h\x01h-\x01i\x03\x01j\x01\xea\0\0\x01@\x01\x04self\xe9\0\0\xeb\0\x04\
\0\x1c[method]incoming-body.stream\x01l\x01i.\x01@\x01\x04this\xc9\0\0\xed\0\x04\
\0\x1c[static]incoming-body.finish\x01n\x01h.\x01i\x09\x01@\x01\x04self\xef\0\0\xf0\
\0\x04\0![method]future-trailers.subscribe\x01q\x01i%\x01k\xf2\0\x01j\x01\xf3\0\x01\
\x1b\x01j\x01\xf4\0\0\x01k\xf5\0\x01@\x01\x04self\xef\0\0\xf6\0\x04\0\x1b[method\
]future-trailers.get\x01w\x01@\x01\x07headers\xc7\0\0\xe2\0\x04\0\x1e[constructo\
r]outgoing-response\x01x\x01h/\x01@\x01\x04self\xf9\0\0+\x04\0%[method]outgoing-\
response.status-code\x01z\x01@\x02\x04self\xf9\0\x0bstatus-code+\0\xd3\0\x04\0)[\
method]outgoing-response.set-status-code\x01{\x01@\x01\x04self\xf9\0\0\xc7\0\x04\
\0![method]outgoing-response.headers\x01|\x01@\x01\x04self\xf9\0\0\xd0\0\x04\0\x1e\
[method]outgoing-response.body\x01}\x01h0\x01i\x05\x01j\x01\xff\0\0\x01@\x01\x04\
self\xfe\0\0\x80\x01\x04\0\x1b[method]outgoing-body.write\x01\x81\x01\x01j\0\x01\
\x1b\x01@\x02\x04this\xcf\0\x08trailers\xf3\0\0\x82\x01\x04\0\x1c[static]outgoin\
g-body.finish\x01\x83\x01\x01h1\x01@\x01\x04self\x84\x01\0\xf0\0\x04\0*[method]f\
uture-incoming-response.subscribe\x01\x85\x01\x01i,\x01j\x01\x86\x01\x01\x1b\x01\
j\x01\x87\x01\0\x01k\x88\x01\x01@\x01\x04self\x84\x01\0\x89\x01\x04\0$[method]fu\
ture-incoming-response.get\x01\x8a\x01\x01h\x07\x01k\x1b\x01@\x01\x03err\x8b\x01\
\0\x8c\x01\x04\0\x0fhttp-error-code\x01\x8d\x01\x03\0\x15wasi:http/types@0.2.2\x05\
\x0e\x02\x03\0\x09\x10outgoing-request\x02\x03\0\x09\x0frequest-options\x02\x03\0\
\x09\x18future-incoming-response\x02\x03\0\x09\x0aerror-code\x01B\x0f\x02\x03\x02\
\x01\x0f\x04\0\x10outgoing-request\x03\0\0\x02\x03\x02\x01\x10\x04\0\x0frequest-\
options\x03\0\x02\x02\x03\x02\x01\x11\x04\0\x18future-incoming-response\x03\0\x04\
\x02\x03\x02\x01\x12\x04\0\x0aerror-code\x03\0\x06\x01i\x01\x01i\x03\x01k\x09\x01\
i\x05\x01j\x01\x0b\x01\x07\x01@\x02\x07request\x08\x07options\x0a\0\x0c\x04\0\x06\
handle\x01\x0d\x03\0\x20wasi:http/outgoing-handler@0.2.2\x05\x13\x02\x03\0\x09\x10\
incoming-request\x02\x03\0\x09\x11response-outparam\x01B\x08\x02\x03\x02\x01\x14\
\x04\0\x10incoming-request\x03\0\0\x02\x03\x02\x01\x15\x04\0\x11response-outpara\
m\x03\0\x02\x01i\x01\x01i\x03\x01@\x02\x07request\x04\x0cresponse-out\x05\x01\0\x04\
\0\x06handle\x01\x06\x04\0\x20wasi:http/incoming-handler@0.2.2\x05\x16\x04\0\x15\
wasi:http/proxy@0.2.2\x04\0\x0b\x0b\x01\0\x05proxy\x03\0\0\0G\x09producers\x01\x0c\
processed-by\x02\x0dwit-component\x070.218.0\x10wit-bindgen-rust\x060.33.0";
        };
    };
}
#[doc(inline)]
pub use __export_proxy_impl as _export_proxy;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.33.0:wasi:http@0.2.2:proxy-with-all-of-its-exports-removed:encoded worldrust-wasi-from-crates-io-proxy-world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 6921] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xed4\x01A\x02\x01A\x1f\
\x01B\x0a\x04\0\x08pollable\x03\x01\x01h\0\x01@\x01\x04self\x01\0\x7f\x04\0\x16[\
method]pollable.ready\x01\x02\x01@\x01\x04self\x01\x01\0\x04\0\x16[method]pollab\
le.block\x01\x03\x01p\x01\x01py\x01@\x01\x02in\x04\0\x05\x04\0\x04poll\x01\x06\x03\
\0\x12wasi:io/poll@0.2.2\x05\0\x02\x03\0\0\x08pollable\x01B\x0f\x02\x03\x02\x01\x01\
\x04\0\x08pollable\x03\0\0\x01w\x04\0\x07instant\x03\0\x02\x01w\x04\0\x08duratio\
n\x03\0\x04\x01@\0\0\x03\x04\0\x03now\x01\x06\x01@\0\0\x05\x04\0\x0aresolution\x01\
\x07\x01i\x01\x01@\x01\x04when\x03\0\x08\x04\0\x11subscribe-instant\x01\x09\x01@\
\x01\x04when\x05\0\x08\x04\0\x12subscribe-duration\x01\x0a\x03\0!wasi:clocks/mon\
otonic-clock@0.2.2\x05\x02\x01B\x05\x01r\x02\x07secondsw\x0bnanosecondsy\x04\0\x08\
datetime\x03\0\0\x01@\0\0\x01\x04\0\x03now\x01\x02\x04\0\x0aresolution\x01\x02\x03\
\0\x1cwasi:clocks/wall-clock@0.2.2\x05\x03\x01B\x05\x01p}\x01@\x01\x03lenw\0\0\x04\
\0\x10get-random-bytes\x01\x01\x01@\0\0w\x04\0\x0eget-random-u64\x01\x02\x03\0\x18\
wasi:random/random@0.2.2\x05\x04\x01B\x04\x04\0\x05error\x03\x01\x01h\0\x01@\x01\
\x04self\x01\0s\x04\0\x1d[method]error.to-debug-string\x01\x02\x03\0\x13wasi:io/\
error@0.2.2\x05\x05\x02\x03\0\x04\x05error\x01B(\x02\x03\x02\x01\x06\x04\0\x05er\
ror\x03\0\0\x02\x03\x02\x01\x01\x04\0\x08pollable\x03\0\x02\x01i\x01\x01q\x02\x15\
last-operation-failed\x01\x04\0\x06closed\0\0\x04\0\x0cstream-error\x03\0\x05\x04\
\0\x0cinput-stream\x03\x01\x04\0\x0doutput-stream\x03\x01\x01h\x07\x01p}\x01j\x01\
\x0a\x01\x06\x01@\x02\x04self\x09\x03lenw\0\x0b\x04\0\x19[method]input-stream.re\
ad\x01\x0c\x04\0\"[method]input-stream.blocking-read\x01\x0c\x01j\x01w\x01\x06\x01\
@\x02\x04self\x09\x03lenw\0\x0d\x04\0\x19[method]input-stream.skip\x01\x0e\x04\0\
\"[method]input-stream.blocking-skip\x01\x0e\x01i\x03\x01@\x01\x04self\x09\0\x0f\
\x04\0\x1e[method]input-stream.subscribe\x01\x10\x01h\x08\x01@\x01\x04self\x11\0\
\x0d\x04\0![method]output-stream.check-write\x01\x12\x01j\0\x01\x06\x01@\x02\x04\
self\x11\x08contents\x0a\0\x13\x04\0\x1b[method]output-stream.write\x01\x14\x04\0\
.[method]output-stream.blocking-write-and-flush\x01\x14\x01@\x01\x04self\x11\0\x13\
\x04\0\x1b[method]output-stream.flush\x01\x15\x04\0$[method]output-stream.blocki\
ng-flush\x01\x15\x01@\x01\x04self\x11\0\x0f\x04\0\x1f[method]output-stream.subsc\
ribe\x01\x16\x01@\x02\x04self\x11\x03lenw\0\x13\x04\0\"[method]output-stream.wri\
te-zeroes\x01\x17\x04\05[method]output-stream.blocking-write-zeroes-and-flush\x01\
\x17\x01@\x03\x04self\x11\x03src\x09\x03lenw\0\x0d\x04\0\x1c[method]output-strea\
m.splice\x01\x18\x04\0%[method]output-stream.blocking-splice\x01\x18\x03\0\x15wa\
si:io/streams@0.2.2\x05\x07\x02\x03\0\x05\x0doutput-stream\x01B\x05\x02\x03\x02\x01\
\x08\x04\0\x0doutput-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x0aget-stdout\x01\
\x03\x03\0\x15wasi:cli/stdout@0.2.2\x05\x09\x01B\x05\x02\x03\x02\x01\x08\x04\0\x0d\
output-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x0aget-stderr\x01\x03\x03\0\x15\
wasi:cli/stderr@0.2.2\x05\x0a\x02\x03\0\x05\x0cinput-stream\x01B\x05\x02\x03\x02\
\x01\x0b\x04\0\x0cinput-stream\x03\0\0\x01i\x01\x01@\0\0\x02\x04\0\x09get-stdin\x01\
\x03\x03\0\x14wasi:cli/stdin@0.2.2\x05\x0c\x02\x03\0\x01\x08duration\x01B\xc1\x01\
\x02\x03\x02\x01\x0d\x04\0\x08duration\x03\0\0\x02\x03\x02\x01\x0b\x04\0\x0cinpu\
t-stream\x03\0\x02\x02\x03\x02\x01\x08\x04\0\x0doutput-stream\x03\0\x04\x02\x03\x02\
\x01\x06\x04\0\x08io-error\x03\0\x06\x02\x03\x02\x01\x01\x04\0\x08pollable\x03\0\
\x08\x01q\x0a\x03get\0\0\x04head\0\0\x04post\0\0\x03put\0\0\x06delete\0\0\x07con\
nect\0\0\x07options\0\0\x05trace\0\0\x05patch\0\0\x05other\x01s\0\x04\0\x06metho\
d\x03\0\x0a\x01q\x03\x04HTTP\0\0\x05HTTPS\0\0\x05other\x01s\0\x04\0\x06scheme\x03\
\0\x0c\x01ks\x01k{\x01r\x02\x05rcode\x0e\x09info-code\x0f\x04\0\x11DNS-error-pay\
load\x03\0\x10\x01k}\x01r\x02\x08alert-id\x12\x0dalert-message\x0e\x04\0\x1aTLS-\
alert-received-payload\x03\0\x13\x01ky\x01r\x02\x0afield-name\x0e\x0afield-size\x15\
\x04\0\x12field-size-payload\x03\0\x16\x01kw\x01k\x17\x01q'\x0bDNS-timeout\0\0\x09\
DNS-error\x01\x11\0\x15destination-not-found\0\0\x17destination-unavailable\0\0\x19\
destination-IP-prohibited\0\0\x19destination-IP-unroutable\0\0\x12connection-ref\
used\0\0\x15connection-terminated\0\0\x12connection-timeout\0\0\x17connection-re\
ad-timeout\0\0\x18connection-write-timeout\0\0\x18connection-limit-reached\0\0\x12\
TLS-protocol-error\0\0\x15TLS-certificate-error\0\0\x12TLS-alert-received\x01\x14\
\0\x13HTTP-request-denied\0\0\x1cHTTP-request-length-required\0\0\x16HTTP-reques\
t-body-size\x01\x18\0\x1bHTTP-request-method-invalid\0\0\x18HTTP-request-URI-inv\
alid\0\0\x19HTTP-request-URI-too-long\0\0\x20HTTP-request-header-section-size\x01\
\x15\0\x18HTTP-request-header-size\x01\x19\0!HTTP-request-trailer-section-size\x01\
\x15\0\x19HTTP-request-trailer-size\x01\x17\0\x18HTTP-response-incomplete\0\0!HT\
TP-response-header-section-size\x01\x15\0\x19HTTP-response-header-size\x01\x17\0\
\x17HTTP-response-body-size\x01\x18\0\"HTTP-response-trailer-section-size\x01\x15\
\0\x1aHTTP-response-trailer-size\x01\x17\0\x1dHTTP-response-transfer-coding\x01\x0e\
\0\x1cHTTP-response-content-coding\x01\x0e\0\x15HTTP-response-timeout\0\0\x13HTT\
P-upgrade-failed\0\0\x13HTTP-protocol-error\0\0\x0dloop-detected\0\0\x13configur\
ation-error\0\0\x0einternal-error\x01\x0e\0\x04\0\x0aerror-code\x03\0\x1a\x01q\x03\
\x0einvalid-syntax\0\0\x09forbidden\0\0\x09immutable\0\0\x04\0\x0cheader-error\x03\
\0\x1c\x01s\x04\0\x09field-key\x03\0\x1e\x04\0\x0afield-name\x03\0\x1f\x01p}\x04\
\0\x0bfield-value\x03\0!\x04\0\x06fields\x03\x01\x04\0\x07headers\x03\0#\x04\0\x08\
trailers\x03\0#\x04\0\x10incoming-request\x03\x01\x04\0\x10outgoing-request\x03\x01\
\x04\0\x0frequest-options\x03\x01\x04\0\x11response-outparam\x03\x01\x01{\x04\0\x0b\
status-code\x03\0*\x04\0\x11incoming-response\x03\x01\x04\0\x0dincoming-body\x03\
\x01\x04\0\x0ffuture-trailers\x03\x01\x04\0\x11outgoing-response\x03\x01\x04\0\x0d\
outgoing-body\x03\x01\x04\0\x18future-incoming-response\x03\x01\x01i#\x01@\0\02\x04\
\0\x13[constructor]fields\x013\x01o\x02\x20\"\x01p4\x01j\x012\x01\x1d\x01@\x01\x07\
entries5\06\x04\0\x18[static]fields.from-list\x017\x01h#\x01p\"\x01@\x02\x04self\
8\x04name\x20\09\x04\0\x12[method]fields.get\x01:\x01@\x02\x04self8\x04name\x20\0\
\x7f\x04\0\x12[method]fields.has\x01;\x01j\0\x01\x1d\x01@\x03\x04self8\x04name\x20\
\x05value9\0<\x04\0\x12[method]fields.set\x01=\x01@\x02\x04self8\x04name\x20\0<\x04\
\0\x15[method]fields.delete\x01>\x01@\x03\x04self8\x04name\x20\x05value\"\0<\x04\
\0\x15[method]fields.append\x01?\x01@\x01\x04self8\05\x04\0\x16[method]fields.en\
tries\x01@\x01@\x01\x04self8\02\x04\0\x14[method]fields.clone\x01A\x01h&\x01@\x01\
\x04self\xc2\0\0\x0b\x04\0\x1f[method]incoming-request.method\x01C\x01@\x01\x04s\
elf\xc2\0\0\x0e\x04\0([method]incoming-request.path-with-query\x01D\x01k\x0d\x01\
@\x01\x04self\xc2\0\0\xc5\0\x04\0\x1f[method]incoming-request.scheme\x01F\x04\0\"\
[method]incoming-request.authority\x01D\x01i$\x01@\x01\x04self\xc2\0\0\xc7\0\x04\
\0\x20[method]incoming-request.headers\x01H\x01i-\x01j\x01\xc9\0\0\x01@\x01\x04s\
elf\xc2\0\0\xca\0\x04\0\x20[method]incoming-request.consume\x01K\x01i'\x01@\x01\x07\
headers\xc7\0\0\xcc\0\x04\0\x1d[constructor]outgoing-request\x01M\x01h'\x01i0\x01\
j\x01\xcf\0\0\x01@\x01\x04self\xce\0\0\xd0\0\x04\0\x1d[method]outgoing-request.b\
ody\x01Q\x01@\x01\x04self\xce\0\0\x0b\x04\0\x1f[method]outgoing-request.method\x01\
R\x01j\0\0\x01@\x02\x04self\xce\0\x06method\x0b\0\xd3\0\x04\0#[method]outgoing-r\
equest.set-method\x01T\x01@\x01\x04self\xce\0\0\x0e\x04\0([method]outgoing-reque\
st.path-with-query\x01U\x01@\x02\x04self\xce\0\x0fpath-with-query\x0e\0\xd3\0\x04\
\0,[method]outgoing-request.set-path-with-query\x01V\x01@\x01\x04self\xce\0\0\xc5\
\0\x04\0\x1f[method]outgoing-request.scheme\x01W\x01@\x02\x04self\xce\0\x06schem\
e\xc5\0\0\xd3\0\x04\0#[method]outgoing-request.set-scheme\x01X\x04\0\"[method]ou\
tgoing-request.authority\x01U\x01@\x02\x04self\xce\0\x09authority\x0e\0\xd3\0\x04\
\0&[method]outgoing-request.set-authority\x01Y\x01@\x01\x04self\xce\0\0\xc7\0\x04\
\0\x20[method]outgoing-request.headers\x01Z\x01i(\x01@\0\0\xdb\0\x04\0\x1c[const\
ructor]request-options\x01\\\x01h(\x01k\x01\x01@\x01\x04self\xdd\0\0\xde\0\x04\0\
'[method]request-options.connect-timeout\x01_\x01@\x02\x04self\xdd\0\x08duration\
\xde\0\0\xd3\0\x04\0+[method]request-options.set-connect-timeout\x01`\x04\0*[met\
hod]request-options.first-byte-timeout\x01_\x04\0.[method]request-options.set-fi\
rst-byte-timeout\x01`\x04\0-[method]request-options.between-bytes-timeout\x01_\x04\
\01[method]request-options.set-between-bytes-timeout\x01`\x01i)\x01i/\x01j\x01\xe2\
\0\x01\x1b\x01@\x02\x05param\xe1\0\x08response\xe3\0\x01\0\x04\0\x1d[static]resp\
onse-outparam.set\x01d\x01h,\x01@\x01\x04self\xe5\0\0+\x04\0\x20[method]incoming\
-response.status\x01f\x01@\x01\x04self\xe5\0\0\xc7\0\x04\0![method]incoming-resp\
onse.headers\x01g\x01@\x01\x04self\xe5\0\0\xca\0\x04\0![method]incoming-response\
.consume\x01h\x01h-\x01i\x03\x01j\x01\xea\0\0\x01@\x01\x04self\xe9\0\0\xeb\0\x04\
\0\x1c[method]incoming-body.stream\x01l\x01i.\x01@\x01\x04this\xc9\0\0\xed\0\x04\
\0\x1c[static]incoming-body.finish\x01n\x01h.\x01i\x09\x01@\x01\x04self\xef\0\0\xf0\
\0\x04\0![method]future-trailers.subscribe\x01q\x01i%\x01k\xf2\0\x01j\x01\xf3\0\x01\
\x1b\x01j\x01\xf4\0\0\x01k\xf5\0\x01@\x01\x04self\xef\0\0\xf6\0\x04\0\x1b[method\
]future-trailers.get\x01w\x01@\x01\x07headers\xc7\0\0\xe2\0\x04\0\x1e[constructo\
r]outgoing-response\x01x\x01h/\x01@\x01\x04self\xf9\0\0+\x04\0%[method]outgoing-\
response.status-code\x01z\x01@\x02\x04self\xf9\0\x0bstatus-code+\0\xd3\0\x04\0)[\
method]outgoing-response.set-status-code\x01{\x01@\x01\x04self\xf9\0\0\xc7\0\x04\
\0![method]outgoing-response.headers\x01|\x01@\x01\x04self\xf9\0\0\xd0\0\x04\0\x1e\
[method]outgoing-response.body\x01}\x01h0\x01i\x05\x01j\x01\xff\0\0\x01@\x01\x04\
self\xfe\0\0\x80\x01\x04\0\x1b[method]outgoing-body.write\x01\x81\x01\x01j\0\x01\
\x1b\x01@\x02\x04this\xcf\0\x08trailers\xf3\0\0\x82\x01\x04\0\x1c[static]outgoin\
g-body.finish\x01\x83\x01\x01h1\x01@\x01\x04self\x84\x01\0\xf0\0\x04\0*[method]f\
uture-incoming-response.subscribe\x01\x85\x01\x01i,\x01j\x01\x86\x01\x01\x1b\x01\
j\x01\x87\x01\0\x01k\x88\x01\x01@\x01\x04self\x84\x01\0\x89\x01\x04\0$[method]fu\
ture-incoming-response.get\x01\x8a\x01\x01h\x07\x01k\x1b\x01@\x01\x03err\x8b\x01\
\0\x8c\x01\x04\0\x0fhttp-error-code\x01\x8d\x01\x03\0\x15wasi:http/types@0.2.2\x05\
\x0e\x02\x03\0\x09\x10outgoing-request\x02\x03\0\x09\x0frequest-options\x02\x03\0\
\x09\x18future-incoming-response\x02\x03\0\x09\x0aerror-code\x01B\x0f\x02\x03\x02\
\x01\x0f\x04\0\x10outgoing-request\x03\0\0\x02\x03\x02\x01\x10\x04\0\x0frequest-\
options\x03\0\x02\x02\x03\x02\x01\x11\x04\0\x18future-incoming-response\x03\0\x04\
\x02\x03\x02\x01\x12\x04\0\x0aerror-code\x03\0\x06\x01i\x01\x01i\x03\x01k\x09\x01\
i\x05\x01j\x01\x0b\x01\x07\x01@\x02\x07request\x08\x07options\x0a\0\x0c\x04\0\x06\
handle\x01\x0d\x03\0\x20wasi:http/outgoing-handler@0.2.2\x05\x13\x04\05wasi:http\
/proxy-with-all-of-its-exports-removed@0.2.2\x04\0\x0b+\x01\0%proxy-with-all-of-\
its-exports-removed\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dwit-compo\
nent\x070.218.0\x10wit-bindgen-rust\x060.33.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
