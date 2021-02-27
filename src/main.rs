use tide::prelude::*;
use tide::http::headers::HeaderValue;
use tide::security::{CorsMiddleware, Origin};
use tch::Tensor;
use image::{DynamicImage, GenericImageView};
use std::sync::Arc;


mod api;
mod www; // Serve the web interface.

const ENCODER_MODEL_PATH:&str = "models/traced_encoder_cpu.pt";
const DECODER_MODEL_PATH:&str = "models/traced_decoder_cpu.pt";
const MODEL_INPUT_WIDTH:u32 = 255;
const MODEL_INPUT_HEIGHT:u32 = 255;
const LATENT_SIZE:u32 = 8192;

#[derive(Clone)]
pub struct State {
	encoder: Arc<tch::CModule>,
	decoder: Arc<tch::CModule>,
}

impl State {
	fn new() -> Self {
		let encoder_model = tch::CModule::load(ENCODER_MODEL_PATH).expect("Failed to find models at expected location: models/traced_*coder_cpu.pt");
		let decoder_model = tch::CModule::load(DECODER_MODEL_PATH).expect("Failed to find models at expected location: models/traced_*coder_cpu.pt");

		State {
			encoder: Arc::new(encoder_model),
			decoder: Arc::new(decoder_model),
		}
	}
}

#[async_std::main]
async fn main() -> tide::Result<()> { //Result<(), std::io::Error>
	tide::log::start();

	//let mut app = tide::new();
	let mut app = tide::with_state(State::new());

	let cors = CorsMiddleware::new()
		.allow_methods("GET, POST, OPTIONS".parse::<HeaderValue>().unwrap())
		.allow_origin(Origin::from("*"))
		.allow_credentials(false);
	app.with(cors);

	/*
	app.at("/api").nest({
		let mut api = tide::new();
		api.at("/hello").get(|_| async { Ok("Hello, world") });
	})
	*/
	app.at("/").get(www::index);
	app.at("/api/image:encode")
		.post(api::encode_image);
	app.listen("127.0.0.1:8080").await?;
	Ok(())
}

fn image_to_tensor(img: &DynamicImage) -> Tensor {
	let img_resized = img.resize_to_fill(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, image::imageops::Nearest);

	Tensor::of_data_size(
		img_resized.as_bytes(),
		&[img_resized.width() as i64, img_resized.height() as i64, 3i64],
		tch::kind::Kind::Int8
	).permute(&[2, 0, 1]) // Convert from WHC to CHW.
}

#[cfg(test)]
mod tests {
	#[test]
	fn sanity() {

	}
}